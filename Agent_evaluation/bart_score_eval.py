from bart_score import BARTScorer

with open('../inputs/Netflix_clean.txt', 'r', encoding='utf-8') as file:
    tweets = file.read()

with open('netflix_testsum.txt', 'r', encoding='utf-8') as file:
    summary = file.read()

scorer = BARTScorer(device='cpu', checkpoint='facebook/bart-large-cnn')
# scorer.load(path='bart.pth')
score = scorer.score([tweets], [summary], batch_size=1)[0]
print("BARTScore:", score)