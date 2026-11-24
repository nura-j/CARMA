import torch
import argparse
from src.dataset import DatasetLoader
from src.load_saved_predictions import load_saved_predictions
from src.model_setup import load_model_tl, load_model_transformers # transformers version 4.46.2
from src.training import fine_tune_transformer
from src.testing import test_model
from src.utils import get_device, set_seed, set_model_name
from src.CAP import cap
from src.dataset import create_synonym_replacement_test_loader
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.cuda.empty_cache()



def parse_arguments():
    """Configure and return parsed arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="GPT2",)
    parser.add_argument("--dataset_name", type=str, default="idm", choices=["sst", "idm"])
    parser.add_argument("--data_path", type=str, default="./data/wordnet_data_definitions.json")
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=6e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default=None, type=int, nargs='+',
                        help='Explicit seeds for intervention runs.')
    parser.add_argument("--stability_start_layer", type=int, default=0)
    parser.add_argument("--stability_end_layer", type=int, default=7)
    parser.add_argument("--mi_start_layer", type=int, default=0)
    parser.add_argument("--mi_end_layer", type=int, default=7)
    parser.add_argument("--stability_weight", type=float, default=0.3)
    parser.add_argument("--mi_weight", type=float, default=0.2)
    parser.add_argument("--mi_n_negatives", type=int, default=5,
                        help="Number of negative token examples per MI positive pair.")
    parser.add_argument('--fast_loss', action='store_true',
                        help='Use the vectorised, numerically stable MI loss during CARMA training.')
    parser.add_argument('--mi_share_negatives', action='store_true',
                        help='With --fast_loss, reuse sampled MI negatives across layers.')
    parser.add_argument("--carma_weight", type=float, default=0.2)
    parser.add_argument("--save_model", action='store_true')
    parser.add_argument('--save_each_epoch', action='store_true',
                        help='Save a compact model checkpoint after every epoch.')
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--calculate_layers_percentage", action='store_true')
    parser.add_argument('--layers_percentage', type=float, default=0.3)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--correct_pred_path', type=str, default=None)
    parser.add_argument('--CAP', action='store_true',
                        help='Run CAP intervention (shorthand for --intervention --intervention_type cap).')
    parser.add_argument('--CAP_start_layer', type=int, default=[1], nargs='+')
    parser.add_argument('--grouping_protocol', type=str, default=['sum',], nargs='+')
    parser.add_argument('--granularity', type=str, default='tw',
                        choices=['tw', 'tp', 'token_to_words', 'token_to_phrases', 'random'])
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--save_test_results', action='store_true')
    parser.add_argument('--intervention', action='store_true')
    parser.add_argument('--num_runs', type=int, default=None,
                        help='Number of intervention runs; uses consecutive seeds from --seed (default: 1).')
    parser.add_argument('--intervention_percentage', type=float, default=0.4)
    parser.add_argument('--intervention_type', type=str, nargs='+', default=None,
                        choices=['synonym', 'cap'],
                        help='Specify one or more intervention types (synonym or cap).')

    args = parser.parse_args()
    if args.CAP:
        args.intervention = True
        if args.intervention_type is None:
            args.intervention_type = ['cap']
        elif 'cap' not in args.intervention_type:
            args.intervention_type.append('cap')
    elif args.intervention_type is None:
        args.intervention_type = ['synonym']

    if args.mi_n_negatives < 1:
        parser.error("--mi_n_negatives must be at least 1")
    if args.mi_share_negatives and not args.fast_loss:
        parser.error('--mi_share_negatives requires --fast_loss')
    if args.save_each_epoch and not args.save_model:
        parser.error('--save_each_epoch requires --save_model')
    if args.num_runs is not None and args.num_runs < 1:
        parser.error('--num_runs must be at least 1')
    if args.seeds is not None:
        if len(set(args.seeds)) != len(args.seeds):
            parser.error('--seeds must not contain duplicates')
        if args.num_runs is not None and args.num_runs != len(args.seeds):
            parser.error('--num_runs must match the number of --seeds')
        args.num_runs = len(args.seeds)
    else:
        args.num_runs = args.num_runs or 1

    return args



def setup_model_and_tokenizer(args, device):
    # """Load model and tokenizer, and optionally load pre-trained weights."""
    if args.CAP or 'cap' in args.intervention_type:
        print('Loading model for CAP...')
        # we load tl model as the other version is not supported for all other models
        model_name = args.model_name
        model_path = args.model_path if args.model_path else None
        path_type = 'HF'
        model, tokenizer = load_model_tl(model_name=model_name, model_path=model_path, path_type=path_type, device=device)
        # TransformerLens enables a BOS token for Pythia by default, whereas
        # Hugging Face fine-tuning and evaluation use Pythia without one.
        if set_model_name(model_name).startswith('pythia'):
            tokenizer.add_bos_token = False
    else:
        # model, tokenizer = load_model_tl(args.model_name, device)
        model, tokenizer = load_model_transformers(args.model_name, device)

        if args.model_path:
            # Load weights on CPU first, then move to the device
            state_dict = torch.load(args.model_path, map_location='cpu')['model_state_dict']
            model.load_state_dict(state_dict)
            # model = model.to(device)
            print(f'Model loaded successfully.')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


def prepare_data_loaders(args, tokenizer):
    """Create data loaders using the DatasetLoader."""
    # model_name = set_model_name(args.model_name)
    dataset_loader = DatasetLoader(
        dataset_name=args.dataset_name,
        path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_length or (124 if args.dataset_name == 'sst' else 58),
        batch_size=args.batch_size,
        model_name=args.model_name
    )
    return dataset_loader.create_data_loader()


def train_model(args, model, tokenizer, train_loader, valid_loader, device, model_name):
    """Train the model."""
    print('Starting training...')
    history, model, model_name, total_training_duration = fine_tune_transformer(
        model,
        tokenizer,
        train_loader,
        valid_loader,
        model_name=model_name,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        device=device,
        stability_start_layer=args.stability_start_layer,
        stability_end_layer=args.stability_end_layer,
        mi_start_layer=args.mi_start_layer,
        mi_end_layer=args.mi_end_layer,
        task_name=args.dataset_name,
        stability_weight=args.stability_weight,
        mi_weight=args.mi_weight,
        mi_n_negatives=args.mi_n_negatives,
        fast_loss=args.fast_loss,
        mi_share_negatives=args.mi_share_negatives,
        carma_weight=args.carma_weight,
        save_model=args.save_model,
        save_each_epoch=args.save_each_epoch,
        save_path=args.save_path,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
    )
    print('Training completed.')
    print('Total training duration:', total_training_duration)
    return model_name


def test_with_intervention(intervention_type, args, model, test_loader, tokenizer, device, model_name, task_name, replacement_percentage = 0.4):
    """Perform testing with interventions, including synonyms and CAP."""
    print('Starting intervention...')
    accuracy, precision, recall, f1, correct_predictions = None, None, None, None, None
    if intervention_type == 'synonym':
        if not args.correct_pred_path:
            raise ValueError('Synonym intervention needs --correct_pred_path')
        print('Loading saved predictions...')
        print('args.correct_pred_path:', args.correct_pred_path)
        print('args.batch_size:', args.batch_size)
        print(f'Dataset name: {task_name}')
        print(f'Model name: {model_name}')
        test_dataloader = load_saved_predictions(args.correct_pred_path, batch_size=args.batch_size)
        test_dataloader = create_synonym_replacement_test_loader(
            original_test_loader=test_dataloader,
            tokenizer=tokenizer,
            model_name=args.model_name,
            dataset_name=task_name,
            replacement_percentage=replacement_percentage,
        )

    if intervention_type == 'synonym':
        # Synonym Replacement Intervention
        print('Performing synonym replacement intervention...')
        accuracy, precision, recall, f1, correct_predictions, save_path, top_k_accuracy = test_model(
            model,
            tokenizer,
            test_dataloader,
            device=device,
            model_name=model_name,
            save_results=args.save_test_results,
            task_name=args.dataset_name,
            intervention_type='synonym',
            seed=args.seed,
            replacement_percentage=replacement_percentage,
            k=3
        )
        return accuracy, precision, recall, f1, correct_predictions, save_path, top_k_accuracy

    elif intervention_type == 'cap':
        # CAP Intervention
        print('Performing CAP intervention...')
        torch.set_grad_enabled(False)
        for start_layer in args.CAP_start_layer:
            for grouping_protocol in args.grouping_protocol:
                print(f'Grouping protocol: {grouping_protocol}')
                print(f'Granularity: {args.granularity}')
                cap(
                    model,
                    tokenizer=tokenizer,
                    test_dataloader=test_loader if not args.correct_pred_path else None,
                    correct_predictions_path=args.correct_pred_path,
                    device=device,
                    model_name=model_name,
                    supervision_type='fine_tuned' if args.model_path else 'original',
                    seed=args.seed,
                    task_type=args.dataset_name,
                    start_layer=start_layer,
                    grouping_protocol=grouping_protocol,
                    granularity=args.granularity,
                    batch_size=args.batch_size,
                )


    else:
        raise ValueError(f"Unknown intervention type: {intervention_type}")
    return accuracy, precision, recall, f1, correct_predictions, None, None


def main():
    args = parse_arguments()
    device = get_device()
    set_seed(args.seed)
    print(f"\nRunning with seed {args.seed} at {device}")
    summary_results_log = {}
    summary_results_log['model_name']={}

    model, tokenizer = setup_model_and_tokenizer(args, device)
    if args.model_path:
        model_name = set_model_name(args.model_path)
        checkpoint = torch.load(args.model_path, map_location='cpu')
        summary_results_log['training_config'] = dict(checkpoint.get('training_config', {}))
        summary_results_log['training_config']['device'] = str(device)

    else:
        model_name = set_model_name(args.model_name)
        summary_results_log['training_config'] = vars(args)
        summary_results_log['training_config']['device'] = device

    summary_results_log['model_name'] = model_name

    # Loaders initialization
    train_loader, valid_loader, test_loader, correct_pred_save_path = None, None, None, None
    args.model_name = set_model_name(args.model_name)
    need_test_loader = args.test or (
        args.intervention and 'cap' in args.intervention_type and not args.correct_pred_path
    )
    if args.train:
        # Create train and validation loaders for training
        if need_test_loader:
            train_loader, valid_loader, test_loader = prepare_data_loaders(args, tokenizer)
            print('Size of test_loader:', len(test_loader))
        else:
            train_loader, valid_loader, _ = prepare_data_loaders(args, tokenizer)
        print('Size of train_loader:', len(train_loader))
        print('Size of valid_loader:', len(valid_loader))
    if need_test_loader and test_loader is None:
        # CAP without saved predictions needs the regular test loader.
        _, _, test_loader = prepare_data_loaders(args, tokenizer)
        print('Size of test_loader:', len(test_loader))

    # if args.test or ('synonym' in args.intervention_type and args.intervention):
    #     # Create test loader for testing or intervention
    #     _, _, test_loader = prepare_data_loaders(args, tokenizer)
    #     print('Size of test_loader:', len(test_loader))

    if args.train:
        print('Starting training...')
        print('Device:', device)
        # check model is on the correct device
        model = model.to(device)
        model_name = train_model(args, model, tokenizer, train_loader, valid_loader, device, model_name=args.model_name)

    if args.test:
        print('Starting testing...')
        accuracy, precision, recall, f1, correct_predictions, correct_pred_save_path, top_k_accuracy = test_model(model=model, tokenizer=tokenizer, test_dataloader=test_loader, device=device, model_name=model_name, save_results=args.save_test_results or (args.intervention and 'synonym' in args.intervention_type and not args.correct_pred_path),
                   task_name=args.dataset_name)
        summary_results_log['test_results'] = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'correct_predictions': correct_predictions}
        if correct_pred_save_path and not args.correct_pred_path:
            args.correct_pred_path = correct_pred_save_path
        # accuracy, precision, recall, f1, correct_predictions, correct_pred_save_path =  test_model(model=model, tokenizer=tokenizer, test_dataloader=test_loader, device=device, model_name=model_name, save_results=args.save_test_results,
        #            task_name=args.dataset_name)
        # print(m)
        # print('Size of test_loader:', len(test_loader))
    if args.intervention:
        print(f'Starting intervention ({args.intervention_type})...')
        summary_results_log['intervention_results'] =  []
        if not args.correct_pred_path:
            print('arg.correct_pred_path is:', args.correct_pred_path)
            print('Updating correct predictions path... to:', correct_pred_save_path) #The one used for testing ~ could be removed  and raise error
            args.correct_pred_path = correct_pred_save_path
        run_seeds = args.seeds if args.seeds is not None else [
            args.seed + run_index for run_index in range(args.num_runs)
        ]
        for seed in run_seeds:
            print(f'Running intervention with seed {seed}')
            set_seed(seed)
            run_args = argparse.Namespace(**vars(args))
            run_args.seed = seed
            for intervention_type in args.intervention_type:
                accuracy, precision, recall, f1, correct_predictions, _, top_k_accuracy = test_with_intervention(
                    intervention_type, run_args, model, test_loader, tokenizer, device,
                    model_name=model_name, task_name=args.dataset_name,
                    replacement_percentage=args.intervention_percentage,
                )
                summary_results_log['intervention_results'].append({
                    intervention_type: {
                        'accuracy': accuracy, 'precision': precision, 'recall': recall,
                        'f1': f1, 'correct_predictions': correct_predictions,
                        'seed': seed, 'intervention_percentage': args.intervention_percentage,
                        'top_k_accuracy': top_k_accuracy,
                    }
                })

    # Save the summary results log
    # print(summary_results_log)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    log_file_path = os.path.join(base_dir, f'results')
    os.makedirs(log_file_path, exist_ok=True)
    file_summary = os.path.join(log_file_path, f'{model_name}_summary_results_.txt')
    with open(file_summary, 'a') as file:
        file.write(str(summary_results_log) + '\n')


if __name__ == "__main__":
    main()

