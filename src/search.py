import hydra
import optuna

from src import OPTUNA_LOCAL_DATABASE
from src.configs import register_configs
from src.configs.search import SearchConfig
from src.train import train


@hydra.main(config_path=None, config_name="search")
def main(cfg: SearchConfig):
    search(cfg=cfg)


def search(cfg: SearchConfig):

    train_config = cfg.train_config
    train_config.max_epochs = cfg.max_epochs

    def objective(trial: optuna.Trial):

        train_config.gamma = trial.suggest_float(name="gamma", low=1, high=10)
        train_config.rnn_units = int(trial.suggest_loguniform(name="rnn_units/4", low=8, high=128)) * 4
        train_config.top_classifier_units = trial.suggest_int(name="top_classifier_units", low=32, high=128)

        trainer, system, datamodule = train(cfg=train_config, trial=trial)
        test_results = trainer.test(system, test_dataloaders=[datamodule.val_dataloader()])
        return test_results[0]

    study = optuna.create_study(
        load_if_exists=True, storage=OPTUNA_LOCAL_DATABASE, direction="maximize", study_name=cfg.study_name
    )
    study.optimize(objective, n_trials=cfg.n_trials)


if __name__ == "__main__":
    register_configs()
    main()
