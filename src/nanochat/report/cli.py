from nanochat import workspace
from nanochat.common import get_dist_info
from nanochat.report.base import BaseReport, Report


def get_report() -> BaseReport:
    # just for convenience, only rank 0 logs to report
    _, ddp_rank, _, _ = get_dist_info()
    if ddp_rank == 0:
        return Report(workspace.report_dir())
    else:
        return BaseReport(workspace.report_dir())


def manage_report(command: str = "generate"):
    match command:
        case "generate":
            get_report().generate()
        case "reset":
            get_report().reset()
        case _:
            raise ValueError(f"Unknown command: {command}")
