import torch
from tqdm import tqdm


def test_classification(net, test_loader, max_iteration=None, description=None):
    total, correct = 0, 0
    max_iteration = len(test_loader) if max_iteration is None else max_iteration

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc=description)

        for i, (inp, target) in enumerate(progress_bar):
            inp, target = inp.cuda(), target.cuda()
            out = net(inp)
            total += inp.size(0)
            correct += torch.sum(out.argmax(1) == target).item()

            progress_bar.set_postfix({"acc": correct / total})

            if i >= max_iteration:
                break

    print(correct / total)
    return correct / total
