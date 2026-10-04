from pathlib import Path


HTML = (Path(__file__).parents[1] / "dashboard" / "index.html").read_text(encoding="utf-8")
REVIEW_HTML = (Path(__file__).parents[1] / "dashboard" / "review.html").read_text(encoding="utf-8")
LID_HTML = (Path(__file__).parents[1] / "dashboard" / "lid.html").read_text(encoding="utf-8")
LONGFORM_HTML = (Path(__file__).parents[1] / "dashboard" / "longform.html").read_text(encoding="utf-8")


def test_html_escape_covers_text_and_attribute_metacharacters():
    assert "replace(/[&<>\"']/g" in HTML
    for entity in ("&amp;", "&lt;", "&gt;", "&quot;", "&#39;"):
        assert entity in HTML


def test_dynamic_ids_errors_and_options_use_escape_helpers():
    assert "function jsArg(x)" in HTML
    assert "onclick=\"detail(${jsArg(r._file)})\"" in HTML
    assert "'⚠ '+escHtml(d.error)" in HTML
    assert '<option value="${escAttr(m)}">' in HTML
    assert "${escHtml(it.id)}" in HTML


def test_chat_audio_template_is_visible_for_mimo_asr():
    assert "'openai-chat-audio'" in HTML
    assert "/v1/chat/completions + input_audio" in HTML


def test_interface_smoke_dialog_uses_backend_action_and_output_labels():
    assert "const action=d.test_action" in HTML
    assert "const outputLabel=d.output_label" in HTML
    assert "${outputLabel}：${d.smoke_text||''}" in HTML
    assert "d.smoke_audio" in HTML
    assert "译后音频 · TTS（本地暂存）" in HTML
    assert '<audio controls preload="metadata"' in HTML


def test_result_detail_labels_source_and_saved_tts_audio_players():
    assert "audioRow('原始音频',amap[s.id])" in HTML
    assert "audioRow('译后音频 · TTS',s.tts_audio)" in HTML
    assert "const refLabel=isTranslate?'参考译文':'参考'" in HTML
    assert "const outputLabel=isTranslate?'模型译文':'识别'" in HTML
    assert "s.source_asr_hyp" in HTML
    assert "源端 ASR" in HTML


def test_asr_detail_diff_uses_the_same_units_as_the_metric():
    assert "function diffHtml(a,b,unit='char')" in HTML
    assert "const diffUnit=errLabel==='WER'?'word':'char'" in HTML
    assert "Math.min(sub" in HTML


def test_gpu_speech_and_translation_metrics_are_visible():
    assert "算 XCOMET-XL" in HTML
    assert 'id="in-speech"' in HTML
    assert "语音同传 · 开启 TTS" in HTML
    assert "speech_eval" in HTML
    assert "r.xcomet" in HTML and "su.xcomet" in HTML
    assert "r.utmos" in HTML and "su.utmos" in HTML
    assert "r.speaker_similarity" in HTML and "su.speaker_similarity" in HTML


def test_longform_results_open_dedicated_timeline_page():
    assert "/api/manifest_longform_ids?manifest=" in HTML
    assert 'href="/longform?file=${encodeURIComponent(DETAIL_CACHE.file)}' in HTML
    assert "打开长音频时间轴" in HTML
    assert "/api/longform_sample?file=" in LONGFORM_HTML
    assert "原文 · SOURCE" in LONGFORM_HTML
    assert "参考 · REFERENCE" in LONGFORM_HTML
    assert "模型译文 · OUTPUT" in LONGFORM_HTML
    assert "audio.currentTime" in LONGFORM_HTML
    assert "历史结果估算时间" in LONGFORM_HTML
    assert "真实分段完成时间" in LONGFORM_HTML
    assert "模型真实分段" in LONGFORM_HTML
    assert "源端 ASR · SOURCE ASR" in LONGFORM_HTML
    assert "timingChips" in LONGFORM_HTML
    assert "返回于" in LONGFORM_HTML


def test_custom_interface_form_accepts_key_environment_variable():
    assert 'id="if-key-env"' in HTML
    assert "key_env:$('#if-key-env').value.trim()" in HTML
    assert "(m.base_url||m.url||'')" in HTML


def test_job_list_exposes_sanitized_request_detail():
    assert "/api/job_request?id=" in HTML
    assert "showJobRequest" in HTML
    assert "任务派发后立即可看契约投影" in HTML
    assert "鉴权、音频路径和内容均隐藏" in HTML
    assert "请求详情" in HTML
    assert "/api/job_response?id=" in HTML
    assert "showJobResponse" in HTML
    assert "返回示例" in HTML


def test_cli_running_job_is_not_misclassified_as_queued():
    assert "(j.stage||'').startsWith('排队中')" in HTML
    assert "j.status=='running'&&!j.started_at" not in HTML
    assert "历史 CLI 无独立日志" in HTML


def test_accuracy_mode_defaults_to_bounded_parallel_workers():
    assert 'id="mode-accuracy"' in HTML
    assert 'id="mode-latency"' in HTML
    assert 'id="in-workers"' in HTML
    assert "RUN_MODE='accuracy'" in HTML
    assert "Math.max(1,Math.min(16" in HTML
    assert "run_mode,workers" in HTML
    assert "精度并发" in HTML
    assert "concurrent_latency_p50_s" in HTML
    assert "并发观测" in HTML


def test_dataset_scatter_can_expand_to_fullscreen_chart():
    assert 'class="chart-zoom"' in HTML
    assert 'id="scatter-fullscreen"' in HTML
    assert 'id="scatter-full"' in HTML
    assert "function openScatterFullscreen()" in HTML
    assert "SCATTER_FULL_CHART.setOption(SCATTER_OPTION,true)" in HTML
    assert "closeScatterFullscreen();detail(p.data[3]+'.json')" in HTML
    assert "if(event.target===this)closeScatterFullscreen()" in HTML


def test_simultaneous_results_prioritize_official_long_yaal_clocks():
    assert "long_yaal_cu_ms" in HTML
    assert "long_yaal_ca_ms" in HTML
    assert "LongYAAL CU" in HTML
    assert "LongYAAL CA" in HTML
    assert "computation-unaware" in HTML
    assert "computation-aware" in HTML


def test_metric_help_covers_text_and_speech_simultaneous_evaluation():
    for key in (
        "COMET", "AL / LAAL", "SoftSegmenter", "Source ASR", "Finish Tail",
        "Stage Latency", "Incremental TTFB", "Incremental Churn", "Revision Rate",
        "Stability Coverage", "TTS CER/WER",
    ):
        assert "{key:'" + key + "'" in HTML
    assert "document_fallback" in HTML
    assert "闪烁量 0 不能解释成" in HTML


def test_simult_run_filters_unsupported_language_pairs_before_dispatch():
    assert "function contractAllowsLanguagePair" in HTML
    assert "contractAllowsLanguagePair(c.model,c.dataset,c.target_lang)" in HTML
    assert "无兼容语向" in HTML
    assert "跳过 ${pairSkipped.length} 个未登记语向" in HTML


def test_stream_progress_distinguishes_partial_revision_and_final_text():
    assert "sp.latest_phase==='partial'?'临时译文'" in HTML
    assert "sp.latest_phase==='revision'?'修订中'" in HTML
    assert "sp.latest_phase==='segment'?'最新定稿'" in HTML


def test_language_is_independent_multi_value_experiment_dimension():
    assert 'id="lang-inline"' in HTML
    assert 'class="lang-help"' in HTML
    assert "LANGS_BY_MODEL" in HTML
    assert "languagesForModel(model)" in HTML
    assert "接口 × 数据集 × language" in HTML
    assert "for(const language of langRuns)" in HTML
    assert "parseLanguageExperiments" in HTML
    assert "split(/[;；\\n]+/)" in HTML
    assert "auto; Chinese,English" in HTML
    assert "showLanguageForm" not in HTML
    assert "showLanguageHelp" not in HTML
    assert 'id="if-asr-language"' not in HTML
    assert 'id="if-asr-itn"' in HTML
    assert 'id="if-asr-timestamps"' in HTML
    assert "return_timestamps" in HTML


def test_run_parameters_are_scoped_to_interface_dataset_combinations():
    assert "RUN_CONFIG_BY_COMBO" in HTML
    assert "runConfigForCombo(model,dataset)" in HTML
    assert "任务参数 · 接口 × 数据集" in HTML
    assert 'class="lang-row run-combo"' in HTML
    assert "target_lang:cfg.target_lang" in HTML
    assert "hotwords:!!cfg.hotwords" in HTML
    assert "hotwords_text:cfg.hotwords_text" in HTML
    assert "managedRequestField(model,'target_lang')" in HTML
    assert "managedRequestField(model,'hotwords')" in HTML
    assert "onRequestParamChange" in HTML
    assert "仅多阶段" in HTML
    assert "多阶段 ASR→MT" in HTML


def test_each_run_combination_exposes_data_driven_parameter_limits():
    assert 'class="param-help"' in HTML
    assert 'class="info-dot">i</span>' in HTML
    assert "renderInterfaceLimits(model,dataset,showLang,showTarget,showHotwords,canHotwords)" in HTML
    assert "langSpec.values" in HTML
    assert "explainFixedLanguage" in HTML
    assert "field.minimum" in HTML and "field.maximum" in HTML
    assert "协议模板" in HTML and "首次使用建议先做接口测试" in HTML


def test_review_page_is_explicitly_non_destructive_and_separates_decisions():
    assert "不修改 manifest、infer、result 或 Original CER" in REVIEW_HTML
    assert 'name="data_decision"' in REVIEW_HTML
    assert 'id="scope-decision"' in REVIEW_HTML
    assert 'id="score-issues"' in REVIEW_HTML
    assert "剩余全部=识别错误" in REVIEW_HTML
    assert 'id="dataset-select"' in REVIEW_HTML and 'id="batch-select"' in REVIEW_HTML
    assert "/api/reviews/sample" in REVIEW_HTML
    assert "infer hyp" in REVIEW_HTML
    assert "reviewed_results" not in REVIEW_HTML
    assert "Reviewed CER" in REVIEW_HTML
    assert "rv.n_excluded" in REVIEW_HTML


def test_lid_page_has_clickable_distribution_and_audio_filtering():
    assert 'href="/lid"' in HTML
    assert "/api/lid/results" in LID_HTML
    assert "/api/lid/result" in LID_HTML
    assert "Prediction Distribution" in LID_HTML
    assert "setLabel(item.label)" in LID_HTML
    assert ".fill{display:block" in LID_HTML
    assert "/api/audio?path=" in LID_HTML
    assert "平均置信度" in LID_HTML
    assert "参考文本：" in LID_HTML
    assert "识别结果：" in LID_HTML
    assert "s.ok?s.display_label:'失败'" in LID_HTML


def test_lid_page_can_launch_track_and_cancel_runs():
    assert 'id="lid-form"' in LID_HTML
    assert 'id="lid-dataset"' in LID_HTML
    assert 'id="lid-url"' in LID_HTML
    assert 'id="lid-experiment"' in LID_HTML
    assert "/api/lid/config" in LID_HTML
    assert "/api/lid/run" in LID_HTML
    assert "/api/jobs" in LID_HTML
    assert "/cancel" in LID_HTML
    assert "fp16 / fp32" in LID_HTML
