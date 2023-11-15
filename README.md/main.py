import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
#pip install pandas
#pip3 install scipy
#pip install scikit-learn

# Открываем файлы
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

ratings_subset = ratings.head(100000) # Здесь указать сколько строк нужно будет учитывать из файла ratings.csv

#Чтобы выводились полностью все столбцы и строки
pd.options.display.width = None
pd.options.display.max_columns = None
pd.options.display.max_colwidth = None

# pip install surprise
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import KNNWithZScore
from surprise import accuracy

# Создание объекта Reader для указания диапазона оценок
reader = Reader(rating_scale=(0.5, 5))

# Объединение данных о рейтингах и жанрах фильмов
data = ratings_subset.merge(movies[['movieId', 'genres']], on='movieId')

# Загрузка данных в формат, поддерживаемый библиотекой surprise
data = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)

# Разделение данных на тренировочный и тестовый наборы
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Использование алгоритма KNNWithZScore для построения модели
sim_options = {'name': 'cosine', 'user_based': False}
model = KNNWithZScore(sim_options=sim_options)
model.fit(trainset)

# Получение рекомендаций для фильма с id=100 (можно заменить на другой id фильма), количество n рекомендованных фильмов
movie_id_to_recommend = 100
movie_inner_id = model.trainset.to_inner_iid(movie_id_to_recommend)
movie_neighbors = model.get_neighbors(movie_inner_id, k=100)

print(movies[movies['movieId'] == movie_id_to_recommend])
# Преобразование inner ids обратно в movie ids
movie_neighbors = [model.trainset.to_raw_iid(inner_id) for inner_id in movie_neighbors]
# Вывод рекомендованных фильмов с учетом жанров
n = 10 # Сколько фильмов нужно вывести
recommended_movies = movies[movies['movieId'].isin(movie_neighbors)]
print(recommended_movies[['movieId', 'title', 'genres']].head(n))
print()

from surprise import SVD

# Создание объекта Reader для указания диапазона оценок
reader = Reader(rating_scale=(0.5, 5))

# Объединение данных о рейтингах и жанрах фильмов
data = ratings_subset.merge(movies[['movieId', 'genres']], on='movieId')

# Загрузка данных в формат, поддерживаемый библиотекой surprise
data = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)

# Разделение данных на тренировочный и тестовый наборы
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Использование алгоритма SVD для построения модели
model = SVD()
model.fit(trainset)

# Получение предсказанных оценок для всех фильмов, которые пользователь не смотрел
user_id_to_predict = 100  # Написать ID конкретного пользователя для рекомендации ему фильмов
user_movies = ratings_subset[ratings_subset['userId'] == user_id_to_predict]['movieId']
movies_to_predict = movies[~movies['movieId'].isin(user_movies)]['movieId']

# Получение предсказаний для фильмов, которые пользователь не смотрел
predictions = []
for movie_id in movies_to_predict:
    # Получение информации о жанрах для фильма
    movie_genres = movies[movies['movieId'] == movie_id]['genres'].values[0]
    prediction = model.predict(user_id_to_predict, movie_id)
    predictions.append({
        'movieId': movie_id,
        'title': movies[movies['movieId'] == movie_id]['title'].values[0],
        'predicted_rating': prediction.est,
        'genres': movie_genres
    })

# Сортировка фильмов по предсказанным оценкам
sorted_predictions = sorted(predictions, key=lambda x: x['predicted_rating'], reverse=True)

# n рекомендованных фильмов
n = 10 # n = int(input())
n_recommendations = n  # Можно изменить на любое значение n от 1 до 20
recommended_movies = pd.DataFrame(sorted_predictions[:n_recommendations])

#Вывод n рекомендованных фильмов пользователю на основании predict_ratings
print('UserId:', user_id_to_predict)
print(recommended_movies[['movieId', 'title', 'genres', 'predicted_rating']])