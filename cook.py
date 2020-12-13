# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas
import numpy as np

from collections import defaultdict
from itertools import chain
from time import sleep

# %% [markdown]
# Читаем данные из преобразованного в csv файла с датасетом

# %%
excel_data_df = pandas.read_csv('test.csv', header=0, skip_blank_lines=True)  # /home/devuser/ROSProj/sandbox/src/tests/hacathon/
excel_data_df.info()


# %%
excel_data_df.head()


# %%
excel_data_df['Массив полей'].unique()


# %%
excel_data_df['Уровень плодородия почв'].unique()

# %% [markdown]
# Создаем константы для дальнейшей работы

# %%
plant_species = {
    'Свекла сах': "1",
    'Ячмень пив': "2",
    'Соя': "3",
    'Пшен оз': "4",
    'Пар': "5"
}

# Соответствие класса названию
class_to_spec = {val: key for key, val in plant_species.items()}

prices = {
    'Свекла сах': 3500.0,
    'Ячмень пив': 12000.0,
    'Соя': 30000.0,
    'Пшен оз': 16000.0,
    'Пар': 0
}

yield_coef = [
    0.8, 1.0, 1.2
]

yield_prev_spec = {
    "Пар": {"Пар": 1, "Соя": 1, "Пшен оз": 1, "Ячмень пив": 1, 'Свекла сах': 1},
    "Соя": {"Пар": 4, "Соя": 2, "Пшен оз": 3, "Ячмень пив": 4, 'Свекла сах': 2},
    "Пшен оз": {"Пар": 6, "Соя": 6, "Пшен оз": 3, "Ячмень пив": 4, 'Свекла сах': 5},
    "Ячмень пив": {"Пар": 4, "Соя": 5, "Пшен оз": 4, "Ячмень пив": 3, 'Свекла сах': 5},
    "Свекла сах": {"Пар": 40, "Соя": 50, "Пшен оз": 60, "Ячмень пив": 60, 'Свекла сах': 30},
}

areas = {
    'Центр': 1, 'Восток': 2, 'Слободка': 3, 'Дальний': 4
}
class_to_areas = {val: key for key, val in areas.items()}

# %% [markdown]
# Немножко преобразуем таблицу для упрощенной работы с ней

# %%
def gerbicid_conver(table, name):
    table[name] = table[name].replace(np.nan, 0)
    table[name] = table[name].replace("Серп", 1)

def spec_to_class(table, name):
    global plant_species
    table[name] = table[name].apply(lambda x: plant_species[x])

def area_to_class(table, name):
    global plant_species
    table[name] = table[name].apply(lambda x: areas[x])


# %%
gerbicid_conver(excel_data_df, "2018 Гербицид")
gerbicid_conver(excel_data_df, "2019 Гербицид")
gerbicid_conver(excel_data_df, "2020 Гербицид")

excel_data_df['Крутизна склона, градусы'] = excel_data_df['Крутизна склона, градусы'].apply(lambda x: x.replace('$', '').replace(',', '.')).astype('float')

spec_to_class(excel_data_df, '2018 Культура')
spec_to_class(excel_data_df, '2019 Культура')
spec_to_class(excel_data_df, '2020 Культура')

area_to_class(excel_data_df, 'Массив полей')



# %% [markdown]
# Посчитаем средние показатели посева за три года

# %%
def calc_means(table):
    vol_2018 = defaultdict(float)
    vol_2019 = defaultdict(float)
    vol_2020 = defaultdict(float)

    for index, row in table.iterrows():
        vol_2018[row['2018 Культура']] += row['Площать, гектары']
        vol_2019[row['2019 Культура']] += row['Площать, гектары']
        vol_2020[row['2020 Культура']] += row['Площать, гектары']

    mean_vol = defaultdict(float)
    for spec in vol_2018.keys():
        for m_d in [vol_2018, vol_2019, vol_2020]:
            mean_vol[spec] += m_d[spec]

    for spec in mean_vol.keys():
        mean_vol[spec] = mean_vol[spec] / 3.0
    return mean_vol, vol_2018, vol_2019, vol_2020

mean_vol = calc_means(excel_data_df)
mean_vol_3_years = mean_vol[0]
print(mean_vol)

# %% [markdown]
# Основные функции генетического алгоритма

# %%
def create_gen(chromosom_dictionary, gen_len):
    return ''.join(np.random.choice(chromosom_dictionary, gen_len))


# %%
def mutation(parent, chromosom_dictionary):
    child = parent[:]
    mutation_p = 1.0 / len(child)
    for i in range(len(child)):
        if np.random.random() < mutation_p:
            new_g = np.random.choice(chromosom_dictionary, 1)
            new_g = list(new_g)[0]
            list1 = list(child)
            list1[i] = new_g
            child = ''.join(list1)
    return child

def crossover(parent1, parent2):
    cross_point = np.random.choice(range(len(parent1[1:-1])), 1)[0]
    child = parent1[:cross_point]
    child += parent2[cross_point: ]
    return child

# %% [markdown]
# Создаем отдельный класс-модель для упрощенной работы с данными

# %%
class Field(object):
    def __init__(self, row, yield_prev_spec, class_to_spec):
        self.yield_prev_spec = yield_prev_spec
        self.class_to_spec = class_to_spec

        self.erosion_coef = 1.0 if row['Крутизна склона, градусы'] < 3 else 0.8

        if row['2018 Гербицид'] == 1:
            self.herbicid_coef = 1.0
        elif row['2019 Гербицид'] == 1:
            self.herbicid_coef = 0.2
        elif row['2020 Гербицид'] == 1:
            self.herbicid_coef = 0.5
        else:
            self.herbicid_coef = 1.0

        self.cluster = row['Массив полей']

        self.harvesting_coef = 1.0 if row['Расстоние до асфальта, км'] <= 1.0 else 0.8
        self.rent_coef = 0.7 if row['Статус собственности на 2020 г'] == 'Истекает срок' else 1.0
        self._weeds_coef =  0.7 if row['Наличие многолетних сорняков в 2020г'] == '+' else 1.0

        self.last_spiec = self.class_to_spec[row['2020 Культура']]
        self.field_volume = row['Площать, гектары']

        self.yield_level = {'Средний': 1.0, 'Высокий': 1.2, 'Низкий': 0.8}
        self._yield_level_coef = self.yield_level[row['Уровень плодородия почв']]

    def erosion_lose(self, g):
        """
        Эррозия
        """
        return self.erosion_coef if g == '1' else 1

    def gerbicid_lose(self, spiec):
        """
        Гербициды
        """
        return self.herbicid_coef if spiec in ['1', '3'] else 1.0

    def harvesting_lose(self, spiec):
        """
        Коэфициент дальности от дороги
        """
        return self.harvesting_coef if spiec == '3' else 1.0

    def rent_risc(self):
        """
        Потери, если истекает аренда
        """
        return self.rent_coef

    def weeds_coef(self):
        """
        Потери при сорняках
        """
        return self._weeds_coef

    def yield_level_coef(self, spiec):
        """
        Уровень плодородия почв
        """
        return self._yield_level_coef

    def get_volume_coef(self, spiec):
        """
        Урожайность культур сильно зависит от культурного растения, возделываемого в предыдущем году (таблица зависимости прилагается)
        """
        cs = self.class_to_spec[spiec]

        return self.yield_prev_spec[cs][self.last_spiec]

    def field_volume_coef(self):
        return self.field_volume

# %% [markdown]
# Функции оценки каждого экземпляра

# %%
def es(val1, val2):
    return np.exp(-np.power(val1-val2, 2)/(val2*val2))


def estimate(gen, fields, class_to_spec, prices, means):
    val = 0
    spiec_vol = defaultdict(int)
    clusters_sp = defaultdict(set)
    for i in range(len(gen)):
        g = gen[i]
        field = fields[i]

        spiec_vol[g] += field.field_volume_coef()
        clusters_sp[field.cluster].add(g)

        spiec = class_to_spec[g]
        price = prices[spiec]
        volume = field.field_volume_coef() * field.get_volume_coef(g)
        yield_c = field.yield_level_coef(g)
        weeds_coef = field.weeds_coef()
        rent_risc = field.rent_risc()
        harvesting_lose = field.harvesting_lose(g)
        gerbicid_lose = field.gerbicid_lose(g)
        erosion_lose = field.erosion_lose(g)
        field_val = yield_c * price * volume * weeds_coef * rent_risc * harvesting_lose * gerbicid_lose * erosion_lose
        val += field_val

        if i != 0:
            if g != gen[i-1] and fields[i-1].cluster == field.cluster:
                val *= 0.9

    c = 1
    for key, val in means.items():
        if spiec_vol[key] != 0:
            c = c * es(spiec_vol[key], val)
        else:
            c = c * 0.0000001

    for key, value in clusters_sp.items():
        if len(value) > 3:
            val = val * 0.5
            break

    return val * c


def calc_overall_income(gen, fields, class_to_spec, prices, means):
    val = 0
    spiec_vol = defaultdict(int)
    clusters_sp = defaultdict(set)
    for i in range(len(gen)):
        g = gen[i]
        field = fields[i]

        spiec_vol[g] += field.field_volume_coef()
        clusters_sp[field.cluster].add(g)

        spiec = class_to_spec[g]
        price = prices[spiec]
        volume = field.field_volume_coef() * field.get_volume_coef(g)
        yield_c = field.yield_level_coef(g)
        weeds_coef = field.weeds_coef()
        rent_risc = field.rent_risc()
        harvesting_lose = field.harvesting_lose(g)
        gerbicid_lose = field.gerbicid_lose(g)
        erosion_lose = field.erosion_lose(g)
        field_val = yield_c * price * volume * weeds_coef * rent_risc * harvesting_lose * gerbicid_lose * erosion_lose
        val += field_val

    return val


def estimate_with_nearest(gen, fields, class_to_spec, prices, means):
    val = 0
    spiec_vol = defaultdict(int)
    clusters_sp = defaultdict(set)
    for i in range(len(gen)):
        g = gen[i]
        field = fields[i]

        spiec_vol[g] += field.field_volume_coef()
        clusters_sp[field.cluster].add(g)

        spiec = class_to_spec[g]
        price = prices[spiec]
        volume = field.field_volume_coef() * field.get_volume_coef(g)
        yield_c = field.yield_level_coef(g)
        weeds_coef = field.weeds_coef()
        rent_risc = field.rent_risc()
        harvesting_lose = field.harvesting_lose(g)
        gerbicid_lose = field.gerbicid_lose(g)
        erosion_lose = field.erosion_lose(g)
        field_val = yield_c * price * volume * weeds_coef * rent_risc * harvesting_lose * gerbicid_lose * erosion_lose
        val += field_val

        if i != 0:
            if g != gen[i-1] and fields[i-1].cluster == field.cluster:
                val *= 0.8
    c = 1
    for key, val in means.items():
        if spiec_vol[key] != 0:
            c = c * es(spiec_vol[key], val)
        else:
            c = c * 0.0000001

    for key, value in clusters_sp.items():
        if len(value) > 3:
            val = val * 0.5
            break

    return val * c


def estimate_nearest(gen, fields):
    val = 1.0
    clusters_sp = defaultdict(set)
    for i in range(len(gen)):
        g = gen[i]
        field = fields[i]
        clusters_sp[field.cluster].add(g)

    for i in range(len(gen)):
        if i != 0:
            if g != gen[i-1] and fields[i-1].cluster == field.cluster:
                val *= 0.8

    
    for key, value in clusters_sp.items():
        if len(value) > 3:
            val = val * 0.5
            break
    return val



# %%
fields_count = excel_data_df.shape[0]
fields_count


# %%
fields = []
for index, row in excel_data_df.iterrows():
    f = Field(row, yield_prev_spec, class_to_spec)
    fields.append(f)

print(len(fields))


# %%
g = create_gen(list(plant_species.values()), fields_count)
print(g)
l = estimate(g, fields, class_to_spec, prices, mean_vol_3_years)
print(l)

# %% [markdown]
# Генетический алгоритм

# %%
best = (-1, '')
best_100 = list()


# %%
iterations = 200
generation_len = 1000
chromosom_dictionary = list(plant_species.values())
generation = [(-1, create_gen(chromosom_dictionary, fields_count)) for _ in range(generation_len)]

for _ in range(iterations):
    new_generation = []
    for g in generation:
        val = g[0]
        gen = g[1]
        if val == -1:
            val = estimate(gen, fields, class_to_spec, prices, mean_vol_3_years)
        new_generation.append((val, gen))
    new_generation = sorted(new_generation, reverse=True)

    if best[0] < new_generation[0][0]:
        best = new_generation[0]

    
    best_100.extend(new_generation)
    best_100 = list(set(best_100))
    best_100 = sorted(best_100, reverse=True)
    best_100 = best_100[: min(100, len(best_100))]
    
    val_sum = sum([p[0] for p in new_generation])
    p = [p[0] / val_sum for p in new_generation]

    generation = []

    for _ in range(generation_len):
        if np.random.random() < 0.2:
            i = np.random.choice(range(generation_len), 1, p)[0]
            new_gen = mutation(new_generation[i][1], chromosom_dictionary)
            v = -1
        elif np.random.random() < 0.2:
            i1 = np.random.choice(range(generation_len), 1, p)[0]
            i2 = np.random.choice(range(generation_len), 1, p)[0]
            new_gen = crossover(new_generation[i1][1], new_generation[i2][1])
            v = -1
        else:         
            i = np.random.choice(range(generation_len), 1, p)[0]   
            new_gen = new_generation[i][1]
            v = new_generation[i][0]
        generation.append((val, new_gen))
    sleep(0.001)



print(best)
print(best_100[0], best_100[-1])


# %%
def print_col(gen, class_to_spec):
    for g in gen:
        print(class_to_spec[g])

print_col(best[1], class_to_spec)
print(calc_overall_income(best[1], fields, class_to_spec, prices, mean_vol_3_years))


# %%
most_pricest = []
for t in best_100:
    q = t[1]
    v = calc_overall_income(q, fields, class_to_spec, prices, mean_vol_3_years)
    most_pricest.append((v, q))

most_pricest = sorted(most_pricest, reverse=True)

most_shortest = []
for t in best_100:
    q = t[1]
    v = estimate_nearest(q, fields)
    most_shortest.append((v, q))

most_shortest = sorted(most_shortest, reverse=True)

most_optimal = []
for t in best_100:
    q = t[1]
    v1 = estimate_nearest(q, fields)
    v2 = calc_overall_income(q, fields, class_to_spec, prices, mean_vol_3_years)
    most_optimal.append((v1 * v2, q))

most_optimal = sorted(most_optimal, reverse=True)

print(best)
print(most_pricest[0], most_shortest[0], most_optimal[0])

# %% [markdown]
# Сохранение в Exel, листы Результаты и Прибыль

# %%
def convert_to_names(gen, class_to_spec):
    converted = []
    for g in gen:
        converted.append(class_to_spec[g])
    return converted 

most_pricest_converted = convert_to_names(most_pricest[0][1], class_to_spec)
most_shortest_converted = convert_to_names(most_shortest[0][1], class_to_spec)
most_optimal_converted = convert_to_names(best[1], class_to_spec)

df1 = pandas.DataFrame([p for p in zip(most_optimal_converted, most_pricest_converted, most_shortest_converted)],
                       columns=['Самый оптимальный', 'Который принесет самую большую прибыль', 'Оптимальный по кластеризации'])

most_pricest_profit = calc_overall_income(most_pricest[0][1], fields, class_to_spec, prices, mean_vol_3_years)
most_shortest_profit = calc_overall_income(most_shortest[0][1],  fields, class_to_spec, prices, mean_vol_3_years)
most_optimal_profit = calc_overall_income(best[1],  fields, class_to_spec, prices, mean_vol_3_years)

df2 = pandas.DataFrame([most_optimal_profit, most_pricest_profit, most_shortest_profit],
                       columns=['Потенциальная прибыль'],
                       index=['Самый оптимальный', 'Который принесет самую большую прибыль', 'Оптимальный по кластеризации'])

with pandas.ExcelWriter('output.xlsx') as writer:  
    df1.to_excel(writer, sheet_name='Результаты')
    df2.to_excel(writer, sheet_name='Прибыль')


# %%



