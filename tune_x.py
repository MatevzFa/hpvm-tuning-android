from pathlib import Path

from predtuner import PipedBinaryApp, config_pylogger

from tuning import *


def config_name(args: TuningArgs):
    return "-".join([
        "conf",
        f"{args.model_id}",
        f"{args.max_iter},{args.qos_tuner_threshold},{args.qos_keep_threshold}",
        f"{args.take_best_n}",
        f"{args.cost_model},{args.qos_model}",
    ])


def main():
    args = tuning_args()
    
    print(args)

    tuning_info = get_model_info(args.model_id)

    assert tuning_info.data_shape[0] % args.batch_size == 0

    tuneset, testset = load_datasets(tuning_info.data_dir, tuning_info.data_shape)
    model = prepare_model(tuning_info.model_factory(), tuning_info.checkpoint)
    tuner_exporter, tuner_binary = compile_tuner_binary(
        model=model,
        tuneset=tuneset, testset=testset,
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
    )

    metadata_file = tuner_exporter.metafile

    # Set up logger to put log file in /tmp
    msg_logger = config_pylogger(output_dir="/tmp", verbose=True)

    # Create a `PipedBinaryApp` that communicates with HPVM bin.
    # "TestHPVMApp" is an identifier of this app (used in logging, etc.) and can be anything.
    # Other arguments:
    #   base_dir: which directory to run binary in (default: the dir the binary is in)
    #   qos_relpath: the name of accuracy file generated by the binary.
    #     Defaults to "final_accuracy". For HPVM apps this shouldn't change.
    #   model_storage_folder: where to put saved P1/P2 models.
    app = PipedBinaryApp(
        "TestHPVMApp",
        tuner_binary,
        metadata_file,
        # Where to serialize prediction models if they are used
        # For example, if you use p1 (see below), this will leave you a
        # tuner_results/vgg16_uci-har/p1.pkl
        # which can be quickly reloaded the next time you do tuning with
        model_storage_folder=Path(args.model_storage_dir) / args.model_id,
    )
    tuner = app.get_tuner()
    tuner.tune(
        # Number of iterations in tuning. In practice, use at least 5000, or 10000.
        max_iter=args.max_iter,
        qos_tuner_threshold=args.qos_tuner_threshold,  # QoS threshold to guide tuner into
        # QoS threshold for which we actually keep the configurations
        qos_keep_threshold=args.qos_keep_threshold,
        # Thresholds are relative to baseline -- baseline_acc - 3.0
        is_threshold_relative=True,
        take_best_n=args.take_best_n,  # Take the best 50 configurations,
        cost_model=args.cost_model or None,  # Use linear performance predictor
        qos_model=args.qos_model or None,  # Use P1 QoS predictor
    )

    conf_name = args.out_config or config_name(args)

    out_plot_file = conf_name + ".pdf"
    out_config_file_all = conf_name + ".all.txt"
    out_config_file_best = conf_name + ".best.txt"

    fig = tuner.plot_configs(show_qos_loss=True, connect_best_points=True)
    fig.savefig(out_plot_file, bbox_inches="tight")
    app.dump_hpvm_configs(tuner.best_configs_prefilter, out_config_file_all)
    app.dump_hpvm_configs(tuner.best_configs, out_config_file_best)


if __name__ == '__main__':
    main()
