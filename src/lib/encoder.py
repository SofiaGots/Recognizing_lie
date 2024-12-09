category = ['angry', 'happy', 'neutral', 'sad']

# def encode(emotion):
#     if emotion in category:
#         return category.index(emotion) / (len(category) - 1)
#     raise ValueError('Unknown emotion')

# def decode(index):
#     if 0 <= index <= 1:
#         scaled_index = index * (len(category) - 1)
#         return category[round(scaled_index)]
#     raise ValueError('Index must be between 0 and 1')


def encode(emotion):
    try:
        return category.index(emotion)
    except ValueError:
        raise ValueError(f'Unknown emotion. Must be one of {category}')

def decode(index):
    try:
        return category[int(index)]
    except (IndexError, ValueError):
        raise ValueError(f'Invalid index. Must be between 0 and {len(category)-1}')