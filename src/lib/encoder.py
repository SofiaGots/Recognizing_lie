category = ['angry', 'happy', 'neutral', 'sad']


def encode(emotion):
    if emotion in category:
        return category.index(emotion) / len(category)
    raise ValueError('Unknown emotion')


def decode(index):
    if index < len(category) - 1:
        return category[int(index * len(category))]
    raise ValueError('Index of emotion out of list')
