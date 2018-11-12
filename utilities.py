import nltk


# cherry pick the best translations based on blue scores

def cherry_pick(records, n, upper_bound=1.0):
    bleus = []

    for en, ch_gr, ch_pd in records:
        bleu = nltk.translate.bleu_score.sentence_bleu(
            [ch_gr], ch_pd)  # caculate BLEU by nltk
        bleus.append(bleu)

    lst = [i for i in range(len(records)) if bleus[i] <= upper_bound]
    lst = sorted(lst, key=lambda i: bleus[i], reverse=True)  # sort by BLEU score

    return [records[lst[i]] for i in range(n)]
