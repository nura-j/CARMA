import random

import torch
import argparse
from src.dataset import DatasetLoader
from src.load_saved_predictions import load_saved_predictions
from src.model_setup import load_model_tl, load_model_transformers # transformers version 4.46.2
from src.training import fine_tune_transformer
# from src.test_2 import test_model
from src.testing import test_model
from src.visualisation import history_plot
from typing import List
from src.utils import get_device, set_seed, set_model_name
from src.CAP import cap
from src.dataset import create_synonym_replacement_test_loader, additional_prompts
from src.load_saved_predictions_2 import create_synonym_replacement_test_loader_json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.cuda.empty_cache()



def parse_arguments():
    """Configure and return parsed arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="GPT2",)
    parser.add_argument("--dataset_name", type=str, default="idm", choices=["sst", "idm"])
    parser.add_argument("--data_path", type=str, default="./data/wordnet_data_definitions.json", choices=["./data/wordnet_data_definitions.json", "../data/wordnet_data_definitions.json","stanfordnlp/sst"])
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=6e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default=[42], type=int, nargs='+', help='Random seed.')
    parser.add_argument("--stability_start_layer", type=int, default=0)
    parser.add_argument("--stability_end_layer", type=int, default=7)
    parser.add_argument("--mi_start_layer", type=int, default=0)
    parser.add_argument("--mi_end_layer", type=int, default=7)
    parser.add_argument("--stability_weight", type=float, default=0.3)
    parser.add_argument("--mi_weight", type=float, default=0.2)
    parser.add_argument("--carma_weight", type=float, default=0.2)
    parser.add_argument("--save_model", action='store_true')
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--calculate_layers_percentage", action='store_true')
    parser.add_argument('--layers_percentage', type=float, default=0.3)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--correct_pred_path', type=str, default=None)
    parser.add_argument('--CAP', action='store_true')
    parser.add_argument('--CAP_start_layer', type=int, default=[1], nargs='+')
    parser.add_argument('--grouping_protocol', type=str, default=['sum',], nargs='+')
    parser.add_argument('--granularity', type=str, default='tw')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--save_test_results', action='store_true')
    parser.add_argument('--intervention', action='store_true')
    parser.add_argument('--num_runs', type=int, default=10)
    parser.add_argument('--intervention_percentage', type=float, default=0.4)
    parser.add_argument('--intervention_type', type=str, nargs='+', default=['synonym',],
                        help='Specify one or more intervention types (e.g., synonym cap random).')

    return parser.parse_args()



def setup_model_and_tokenizer(args, device):
    # """Load model and tokenizer, and optionally load pre-trained weights."""
    if args.CAP or 'cap' in args.intervention_type:
        print('Loading model for CAP...')
        # we load tl model as the other version is not supported for all other models
        model_name = args.model_name
        model_path = args.model_path if args.model_path else None
        path_type = 'HF'
        model, tokenizer = load_model_tl(model_name=model_name, model_path=model_path, path_type=path_type, device=device)
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
    tokenizer.add_special_tokens(False)

    return model, tokenizer


def prepare_data_loaders(args, tokenizer):
    """Create data loaders using the DatasetLoader."""
    # model_name = set_model_name(args.model_name)
    dataset_loader = DatasetLoader(
        dataset_name=args.dataset_name,
        path=args.data_path,
        tokenizer=tokenizer,
        max_length=124 if args.dataset_name == 'sst' else 58,
        batch_size=args.batch_size,
        model_name=args.model_name
    )
    return dataset_loader.create_data_loader()


def train_model(args, model, tokenizer, train_loader, valid_loader, device, model_name):
    """Train the model and plot history."""
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
        carma_weight=args.carma_weight,
        save_model=args.save_model,
        save_path=args.save_path,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
    )
    history_plot(history, args.model_name, args.dataset_name)
    print('Training completed.')
    print('Total training duration:', total_training_duration)
    return model_name


def test_with_intervention(intervention_type, args, model, test_loader, tokenizer, device, model_name, task_name, replacement_percentage = 0.4):
    """Perform testing with interventions, including synonyms and CAP."""
    print('Starting intervention...')
    accuracy, precision, recall, f1, correct_predictions, top_k = None, None, None, None, None, None
    if args.correct_pred_path:
        print('Loading saved predictions...')
        print('args.correct_pred_path:', args.correct_pred_path)
        print('args.batch_size:', args.batch_size)
        print(f'Dataset name: {task_name}')
        print(f'Model name: {model_name}')
        test_dataloader = load_saved_predictions(args.correct_pred_path, batch_size=args.batch_size)
        print('test_dataloader for original:', tokenizer.decode(next(iter(test_dataloader))['input_ids'][1]))
        test_dataloader = create_synonym_replacement_test_loader_json(original_loader=test_dataloader, tokenizer=tokenizer, model_name=model_name, dataset_name=task_name, prompts=additional_prompts, replacement_percentage =replacement_percentage)
        print('test_dataloader for synonym:', tokenizer.decode(next(iter(test_dataloader))['input_ids'][1]))

    if intervention_type == 'synonym':
        # Synonym Replacement Intervention
        print('Performing synonym replacement intervention...')
        # synonym_test_loader = create_synonym_replacement_test_loader( # todo fix the creation of the synonym replacement test loader
        #     original_test_loader=test_loader,
        #     tokenizer=tokenizer,
        #     model_name=model_name,
        #     dataset_name=task_name,
        #     replacement_percentage=0.2  # Replace 20% of content with synonyms
        # )

        # accuracy, precision, recall, f1, correct_predictions, _ = test_model(
        #     model,
        #     tokenizer,
        #     test_dataloader,
        #     device=device,
        #     model_name=model_name,
        #     save_results=args.save_test_results,
        #     task_name=args.dataset_name,
        #     intervention_type='synonym',
        #     seed=args.seed,
        #     replacement_percentage=replacement_percentage
        # )
        accuracy, precision, recall, f1, correct_predictions, _ , top_k_accuracy= test_model(
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
        return accuracy, precision, recall, f1, correct_predictions, _, top_k_accuracy
        # return accuracy, precision, recall, f1, correct_predictions

    elif intervention_type == 'cap':
        # CAP Intervention
        print('Performing CAP intervention...')
        torch.set_grad_enabled(False)
        for start_layer in args.CAP_start_layer:
            for grouping_protocol in args.grouping_protocol:
                print(f'Grouping protocol: {grouping_protocol}')
                for granularity in args.granularity:
                    print(f'Granularity: {granularity}')
                    if args.correct_pred_path:
                        cap(
                            model,
                            tokenizer=tokenizer,
                            test_dataloader= None,
                            correct_predictions_path=args.correct_pred_path,
                            device=device,
                            model_name=model_name,
                            supervision_type='fine_tuned',
                            seed=args.seed,
                            task_type=args.dataset_name,
                            # start_layer=args.CAP_start_layer,
                            start_layer=start_layer,
                            # grouping_protocol=args.grouping_protocol,
                            grouping_protocol=grouping_protocol,
                            # granularity=args.granularity,
                            granularity=granularity,
                            batch_size=args.batch_size
                        )
                    else:
                        cap(
                            model,
                            tokenizer=tokenizer,
                            test_dataloader=test_loader,
                            device=device,
                            model_name=model_name,
                            supervision_type='original',
                            seed=args.seed,
                            task_type=args.dataset_name,
                            # start_layer=args.CAP_start_layer,
                            start_layer=start_layer,
                            # grouping_protocol=args.grouping_protocol,
                            grouping_protocol=grouping_protocol,
                            # granularity=args.granularity,
                            granularity=granularity,
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
        summary_results_log['training_config'] = torch.load(args.model_path, map_location=torch.device('cpu'))['training_config']
        summary_results_log['training_config']['device'] = 'cuda' if summary_results_log['training_config'][
                                                                         'device'] == "device(type='cuda')" else 'cpu'

    else:
        model_name = set_model_name(args.model_name)
        summary_results_log['training_config'] = vars(args)
        summary_results_log['training_config']['device'] = device

    summary_results_log['model_name'] = model_name

    # Loaders initialization
    train_loader, valid_loader, test_loader, correct_pred_save_path = None, None, None, None
    args.model_name = set_model_name(args.model_name)
    if args.train:
        # Create train and validation loaders for training
        if args.test or ('synonym' in args.intervention_type and args.intervention):
            train_loader, valid_loader, test_loader = prepare_data_loaders(args, tokenizer)
            print('Size of test_loader:', len(test_loader))
        else:
            train_loader, valid_loader, _ = prepare_data_loaders(args, tokenizer)
        print('Size of train_loader:', len(train_loader))
        print('Size of valid_loader:', len(valid_loader))
    if args.test or ('synonym' in args.intervention_type and args.intervention) and not test_loader:
        # Create test loader for testing only
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
        accuracy, precision, recall, f1, correct_predictions, correct_pred_save_path, top_k_accuracy = test_model(model=model, tokenizer=tokenizer, test_dataloader=test_loader, device=device, model_name=model_name, save_results=args.save_test_results,
                   task_name=args.dataset_name)
        summary_results_log['test_results'] = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'correct_predictions': correct_predictions}
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
        # seeds = [random.randint(1, 10000) for _ in range(args.num_runs)]
        # seeds = [32, 101, 27, 40, 1]
        # for seed in seeds:
        #     print(f"\nRunning with seed {seed}")
        #     set_seed(seed)  # Set the new seed

        for intervention_type in args.intervention_type:
            accuracy, precision, recall, f1, correct_predictions, _, top_k_accuracy= test_with_intervention(intervention_type, args, model, test_loader, tokenizer, device, model_name=model_name, task_name=args.dataset_name, replacement_percentage=args.intervention_percentage)
            # accuracy, precision, recall, f1, correct_predictions = test_with_intervention(intervention_type, args, model, test_loader, tokenizer, device, model_name=model_name, task_name=args.dataset_name)

            summary_results_log['intervention_results'].append({intervention_type: {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'correct_predictions': correct_predictions, 'seed': args.seed, 'intervention_percentage': args.intervention_percentage, 'top_k_accuracy': top_k_accuracy}})

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

