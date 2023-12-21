# final_project_skillbox
Данный репозиторий содержит финальную работу по курсу «Введение в Data Science», специализация: Machine Learning.
Автор: Ольга Красовская (Силюк)

В работе обучается модель машинного обучения для предсказания совершения целевого действия на сайте «СберАвтоподписка».

Для запуска пайплайна обучения потребуется положить в папку data файлы ga_hits.csv и ga_sessions.csv, которые можно взять из https://drive.google.com/drive/folders/1rA4o6KHH-M2KMvBLHp5DZ5gioF2q7hZw

Обученная в пайплайне модель sber_auto_pipe.pkl имеет значение roc_auc_score = 0.69.

Наиболее информативными признаками являются день года и время сеанса, номер сеанса, широта-долгота города, откуда производится сеанс, а также диагональ экрана. Из категориальных признаков наиболее важным является канал привлечения (utm_source).

Для того, чтобы воспользоваться моделью, необходимо заполнить о пользователе следующую информацию:

session_id — ID визита;
client_id — ID посетителя;
visit_date — дата визита;
visit_time — время визита;
visit_number — порядковый номер визита клиента;
utm_source — канал привлечения;
utm_medium — тип привлечения;
utm_campaign — рекламная кампания;
device_category — тип устройства;
device_brand — марка устройства;
device_screen_resolution — разрешение экрана;
device_browser — браузер;
geo_country — страна;
geo_city — город

Пример файла о пользователе содержится в "data\example.json"
