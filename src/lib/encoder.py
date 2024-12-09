category = ['angry', 'happy', 'neutral', 'sad']

# Кодировка эмоции по ее индексу для более удобной дальнейшей работы
def encode(emotion):
    try:
        return category.index(emotion)
    except ValueError:
        raise ValueError(f'Unknown emotion. Must be one of {category}')

# Раскодировка (по индексу получить эмоцию) эмоции
def decode(index):
    try:
        return category[int(index)]
    except (IndexError, ValueError):
        raise ValueError(f'Invalid index. Must be between 0 and {len(category)-1}')