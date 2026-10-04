#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 libspeex(ctypes) 解码讯飞 speex-wb 裸帧 -> 16k/16bit/mono PCM。
讯飞 speex-wb 是标准宽带 speex 码流拼接，解码器靠码流内的 mode 位自分帧。"""
import ctypes
import ctypes.util
import os
import sys

SPEEX_MODEID_WB = 1
SPEEX_GET_FRAME_SIZE = 3
SPEEX_SET_ENH = 0


def find_speex_library():
    """查找可加载的 libspeex；兼容 Linux 运行镜像与 macOS 本地开发。"""
    candidates = [
        os.environ.get("SPEEX_LIBRARY"),
        ctypes.util.find_library("speex"),
        "/usr/lib/x86_64-linux-gnu/libspeex.so.1",
        "/usr/lib/aarch64-linux-gnu/libspeex.so.1",
        "/opt/homebrew/opt/speex/lib/libspeex.dylib",
        "/usr/local/opt/speex/lib/libspeex.dylib",
    ]
    for path in dict.fromkeys(p for p in candidates if p):
        try:
            ctypes.CDLL(path)
            return path
        except OSError:
            continue
    return ""


def decode(speex_bytes: bytes) -> bytes:
    lib = find_speex_library()
    if not lib:
        raise RuntimeError("未找到可加载的 libspeex；Linux 安装 libspeex1，macOS 安装 speex")
    sp = ctypes.CDLL(lib)
    vp, i, ip = ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p
    # 必须给所有指针参数/返回值声明类型，否则 arm64 上指针被截断成32位 -> segfault
    sp.speex_lib_get_mode.restype = vp
    sp.speex_lib_get_mode.argtypes = [i]
    sp.speex_decoder_init.restype = vp
    sp.speex_decoder_init.argtypes = [vp]
    sp.speex_bits_init.argtypes = [vp]
    sp.speex_bits_read_from.argtypes = [vp, vp, i]
    sp.speex_decoder_ctl.argtypes = [vp, i, vp]
    sp.speex_decode_int.argtypes = [vp, vp, vp]
    sp.speex_decode_int.restype = i
    sp.speex_bits_remaining.argtypes = [vp]
    sp.speex_bits_remaining.restype = i
    sp.speex_decoder_destroy.argtypes = [vp]
    sp.speex_bits_destroy.argtypes = [vp]

    mode = sp.speex_lib_get_mode(SPEEX_MODEID_WB)
    state = sp.speex_decoder_init(mode)

    # SpeexBits 结构 < 64B，给足缓冲只传指针即可
    bits = ctypes.create_string_buffer(256)
    sp.speex_bits_init(bits)

    enh = ctypes.c_int(1)
    sp.speex_decoder_ctl(state, SPEEX_SET_ENH, ctypes.byref(enh))
    fsz = ctypes.c_int(0)
    sp.speex_decoder_ctl(state, SPEEX_GET_FRAME_SIZE, ctypes.byref(fsz))
    frame_size = fsz.value or 320

    # 讯飞分帧: [1字节长度L][L字节 speex-wb 帧]，每帧独立解码
    out = (ctypes.c_int16 * frame_size)()
    pcm = bytearray()
    frames = 0
    off, n = 0, len(speex_bytes)
    while off < n:
        L = speex_bytes[off]; off += 1
        if L == 0 or off + L > n:
            break
        frame = speex_bytes[off:off+L]; off += L
        buf = ctypes.create_string_buffer(frame, L)
        sp.speex_bits_read_from(bits, buf, L)
        ret = sp.speex_decode_int(state, bits, out)
        if ret != 0:
            sys.stderr.write(f"[speex] 帧{frames} 解码ret={ret}\n")
            continue
        pcm += bytes(out)
        frames += 1
    sp.speex_decoder_destroy(state)
    sp.speex_bits_destroy(bits)
    sys.stderr.write(f"[speex] {len(speex_bytes)}B -> {frames}帧 x{frame_size} = {len(pcm)}B PCM\n")
    return bytes(pcm)

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "/tmp/xf_tts_out.raw"
    dst = sys.argv[2] if len(sys.argv) > 2 else "/tmp/xf_tts_out.pcm"
    pcm = decode(open(src, "rb").read())
    open(dst, "wb").write(pcm)
    print(f"PCM -> {dst} ({len(pcm)}B, {len(pcm)/32000:.2f}s @16k)")
