from .arguments import get_args
from .checkpointing import load_checkpoint_for_inference
from .enums import Mode
from .utils import ProcessGroupManager, run_rank_n

from pathlib import Path

def main() -> None:
    """main program"""

    mode = Mode.unsharding

    args = get_args(mode)

    if args.unsharded_path is None:
        args.unsharded_path = Path(args.load_args.load_path) / "dolomite_compatible"

    model, _, state_dict = load_checkpoint_for_inference(args, mode, use_meta=True)
    run_rank_n(model.save_pretrained, barrier=ProcessGroupManager.is_initialized())(
        args.unsharded_path if args.load_args.iteration is None else Path(args.unsharded_path) / f"step{args.load_args.iteration}", 
        state_dict=state_dict
    )

    ProcessGroupManager.destroy_process_groups()


if __name__ == "__main__":
    main()
