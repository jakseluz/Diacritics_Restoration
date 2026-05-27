from diacritics_restoration.utils.restorer import DiacriticsRestorer
from diacritics_restoration.utils.dataset import DiacriticsDataset, get_wikipedia_data

import torch
from torch.utils.data import DataLoader


def evaluate_restorer_on_articles(restorer: DiacriticsRestorer, num_articles=1000, batch_size=256):
    print(
        f"=== Evaluating on {num_articles} articles with batch size {batch_size}... ==="
    )
    dataset = DiacriticsDataset(get_wikipedia_data(num_articles=num_articles)["text"].tolist())
    print("Dataset size:", len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size)

    device = restorer.device
    model = restorer.model
    pad_id = dataset.processor.pad_token_id

    print("Running predictions on the dataset...")
    pred_ids, true_ids, conf = predict_on_loader(
        model=model, dataloader=dataloader, device=device, return_confidence=True
    )

    pred_texts = decode_batch(dataset.processor, pred_ids[:3])
    true_texts = decode_batch(dataset.processor, true_ids[:3])

    diacritics = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")
    all_diacritics = 0
    differences_diacritics = 0

    for i, (predicted, expected) in enumerate(zip(pred_texts, true_texts)):
        differences_char = sum(1 for p, t in zip(predicted, expected) if p != t)
        differences_diacritics += sum(
            1 for p, t in zip(predicted, expected) if p != t and t in diacritics
        )
        all_diacritics += sum(1 for t in expected if t in diacritics)
        print(f"\nExample {i}")
        print("PRED:", predicted)
        print("TRUE:", expected)
        print(f"Differences: {differences_char} / {len(expected)}")
        print(
            f"Differences (diacritics): {differences_diacritics} / {all_diacritics} (so far in all examples)"
        )

    print("\n=== Overall Metrics on the dataset ===")
    mask = true_ids != pad_id
    total_differences = (pred_ids[mask] != true_ids[mask]).sum().item()
    total_chars = int(mask.sum().item())
    cer = total_differences / total_chars if total_chars > 0 else 0.0

    is_diacritic = torch.zeros(dataset.processor.vocab_size, dtype=torch.bool)
    for char in diacritics:
        is_diacritic[dataset.processor.char_id[char]] = True

    diacritic_mask = mask & is_diacritic[true_ids]
    all_diacritics = int(diacritic_mask.sum().item())
    differences_diacritics = int(
        (pred_ids[diacritic_mask] != true_ids[diacritic_mask]).sum().item()
    )
    der = differences_diacritics / all_diacritics if all_diacritics > 0 else 0.0

    print(
        f"\nOverall Character Error Rate: {cer:.6f} ({total_differences} differences out of {total_chars} characters)",
        f"\nOverall Diacritics Error Rate: {der:.6f} ({differences_diacritics} differences out of {all_diacritics} diacritics)",
    )


def predict_on_loader(model, dataloader: DataLoader, device, *, return_confidence: bool = False):
    model.eval()
    all_pred_ids = []
    all_true_ids = []
    all_conf = []

    with torch.inference_mode():
        for x_ids, y_ids in dataloader:
            x_ids = x_ids.to(device, non_blocking=True)
            y_ids = y_ids.to(device, non_blocking=True)

            logits = model(x_ids)
            pred_ids = logits.argmax(dim=1)
            all_pred_ids.append(pred_ids.cpu())
            all_true_ids.append(y_ids.cpu())

            if return_confidence:
                probs = torch.softmax(logits, dim=1)
                conf, pred_ids = probs.max(dim=1)
                all_conf.append(conf.cpu())

    pred_ids = torch.cat(all_pred_ids, dim=0)
    true_ids = torch.cat(all_true_ids, dim=0)

    if return_confidence:
        conf = torch.cat(all_conf, dim=0)
        return pred_ids, true_ids, conf
    return pred_ids, true_ids


def decode_batch(processor, ids_2d: torch.Tensor):
    """ids_2d: [N, T] on CPU. Returns list[str] length N.
    sequence_to_text removes <PAD> automatically in your processor.
    """
    return [processor.sequence_to_text(seq.tolist()) for seq in ids_2d]
