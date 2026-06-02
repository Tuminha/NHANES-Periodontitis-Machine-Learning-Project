import matplotlib

matplotlib.use("Agg")

from src.evaluation import export_metrics_table, plot_confusion_matrix


def test_plot_confusion_matrix_handles_count_annotations():
    plot_confusion_matrix([0, 1, 0, 1], [0, 1, 1, 1])


def test_plot_confusion_matrix_handles_normalized_annotations():
    plot_confusion_matrix([0, 1, 0, 1], [0, 1, 1, 1], normalize=True)


def test_export_metrics_table_writes_markdown_without_tabulate(tmp_path):
    output = tmp_path / "metrics.csv"

    export_metrics_table({"model": {"roc_auc": 0.717245, "brier": 0.200252}}, str(output))

    assert output.exists()
    assert (tmp_path / "metrics.md").read_text(encoding="utf-8").startswith("|")
