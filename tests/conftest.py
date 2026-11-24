"""
tests/conftest.py
=================
The trained models and the processed order table are NOT in git (see the README, "Quick start"): they are built by `python run_all.py`
or restored from demo_artifacts.zip.  On a checkout without them (for example a GitHub Actions runner) the tests listed below cannot
run, so they are SKIPPED with an explicit reason instead of failing.  Everything else runs against the tracked metrics files.

tests/test_repo_hygiene.py::test_artifact_skip_list_names_real_tests keeps this list honest: every name must be a real test.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ARTEFACTS = ["models/final_model.joblib", "models/final_calibrator.joblib", "models/preprocessor.joblib", "data/processed/orders.parquet"]
MISSING = [a for a in ARTEFACTS if not (ROOT / a).exists()]

# "<test file>::<test name incl. parameters>" of every test that loads the models or the processed table
NEEDS_ARTEFACTS = {
    "test_app_pages.py::test_batch_labelled_sample_shows_warning_late_labels_and_tier_by_mode",
    "test_app_pages.py::test_batch_labels_missing_columns_and_template",
    "test_app_pages.py::test_every_page_renders_cleanly[batch]",
    "test_app_pages.py::test_every_page_renders_cleanly[explainability]",
    "test_app_pages.py::test_every_page_renders_cleanly[impact]",
    "test_app_pages.py::test_every_page_renders_cleanly[performance]",
    "test_app_pages.py::test_every_page_renders_cleanly[real_orders]",
    "test_app_pages.py::test_every_page_renders_cleanly[score_order]",
    "test_app_pages.py::test_explainability_percentage_is_computed_and_has_howto",
    "test_app_pages.py::test_impact_has_mode_multiselect_sensitivity_and_clean_money",
    "test_app_pages.py::test_performance_comparison_chart_renders_and_lift_note",
    "test_app_pages.py::test_presets_equal_real_test_rows_and_real_orders_page",
    "test_app_pages.py::test_real_orders_confusion_summary_matches_flags",
    "test_app_pages.py::test_real_orders_filters_cascade_and_reset",
    "test_app_pages.py::test_score_page_info_level_note_and_flag_text",
    "test_ceiling_and_challengers.py::test_split_matches_02_train",
    "test_consistency.py::test_flag_equals_high_tier_in_predict",
    "test_dashboard_fixes.py::test_batch_demo_caption_and_small_tier_chart_hidden",
    "test_dashboard_fixes.py::test_country_names_map_to_english_for_display_only",
    "test_dashboard_fixes.py::test_high_and_medium_presets_have_different_payment_types",
    "test_dashboard_fixes.py::test_impact_caption_is_dynamic_and_first_class_warns",
    "test_dashboard_fixes.py::test_per_mode_flagging_totals_equal_test_totals",
    "test_dashboard_fixes.py::test_performance_page_shows_the_table_and_data_built_sentence",
    "test_dashboard_fixes.py::test_presets_load_their_own_row_and_clear_custom_state",
    "test_dashboard_fixes.py::test_untouched_preset_scores_like_the_real_row",
    "test_operating_points.py::test_shipped_artefacts_reproduce_card_auc_and_report",
    "test_pipeline.py::test_chrono_split_no_overlap",
    "test_pipeline.py::test_history_features_no_future_leakage",
    "test_pipeline.py::test_no_forbidden_features_in_model",
    "test_pipeline.py::test_score_order_returns_valid_output",
    "test_repo_hygiene.py::test_demo_bundle_lists_exactly_what_the_app_needs_and_stays_out_of_git",
    "test_report_numbers.py::test_score_order_caption_names_preset_versus_custom_with_exact_scores",
    "test_reproducibility.py::test_calibrated_scores_are_monotone_in_the_raw_scores",
    "test_reproducibility.py::test_scoring_the_same_200_test_orders_twice_gives_identical_scores",
    "test_reproducibility.py::test_split_sizes_and_dates_equal_the_metrics_file",
    "test_rigour.py::test_rule_flags_match_a_brute_force_recomputation_on_200_orders",
    "test_rigour.py::test_stored_rule_baseline_equals_a_brute_force_count_on_the_whole_test_set",
    "test_ui_components.py::test_tier_is_computed_from_the_unrounded_score",
    "test_ui_state.py::test_custom_then_preset_then_custom_carries_no_stale_value",
    "test_ui_state.py::test_each_preset_loads_every_widget_from_its_own_holdout_row[High]",
    "test_ui_state.py::test_each_preset_loads_every_widget_from_its_own_holdout_row[Low]",
    "test_ui_state.py::test_each_preset_loads_every_widget_from_its_own_holdout_row[Medium]",
    "test_ui_state.py::test_impact_caption_and_breakeven_quote_the_current_other_modes_slider[0]",
    "test_ui_state.py::test_impact_caption_and_breakeven_quote_the_current_other_modes_slider[23]",
    "test_ui_state.py::test_impact_caption_and_breakeven_quote_the_current_other_modes_slider[7]",
    "test_ui_state.py::test_impact_default_selection_excludes_first_class",
    "test_ui_state.py::test_impact_selecting_first_class_shows_the_warning_and_removing_it_clears_it",
    "test_ui_state.py::test_payment_reason_codes_use_the_same_type_as_the_widget_in_every_preset",
    "test_ui_state.py::test_presets_are_the_documented_holdout_orders",
}


def pytest_collection_modifyitems(config, items):
    if not MISSING:
        return
    skip = pytest.mark.skip(reason="model artefacts not present (" + ", ".join(MISSING) + "): run python run_all.py or unzip demo_artifacts.zip")
    for item in items:
        if f"{item.path.name}::{item.name}" in NEEDS_ARTEFACTS:
            item.add_marker(skip)
