import matplotlib
import pandas as pd
import numpy as np
from scipy.special import erf
from scipy.stats import norm, skew, kurtosis, chi2, f_oneway
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

matplotlib.use('TkAgg')


# Функция для удаления выбросов
def error_detection(data, q=0.01):
    while True:
        M = np.mean(data)
        S = np.std(data)
        alldone = True

        # Проверяем максимальное значение
        idx_max = np.argmax(data)
        val_max = data[idx_max]
        P_max = 1 - erf((val_max - M) / (S * np.sqrt(2)))  # функция Лапласа
        if P_max < q:
            print(f"индекс: {idx_max} значение: {val_max}")
            data = np.delete(data, idx_max)
            alldone = False

        # Проверяем минимальное значение
        idx_min = np.argmin(data)
        val_min = data[idx_min]
        P_min = 1 - erf((M - val_min) / (S * np.sqrt(2)))
        if P_min < q:
            print(f"индекс: {idx_min} значение: {val_min}")
            data = np.delete(data, idx_min)
            alldone = False

        if alldone:
            break

    return data


# Функция для построения графиков
def graph_data(data, time, plot_name, data_name):
    mean = np.mean(data)
    std = np.std(data)
    centered = data - mean
    scaled = centered / std
    plt.figure(data_name)
    plt.plot(time, centered, label=f"Центрированные данные {data_name}")
    plt.plot(time, scaled, label=f"Масштабированные данные {data_name}")
    plt.title(f"{plot_name}")
    plt.xlabel("Время")
    plt.ylabel("Значения")
    plt.legend()
    plt.show()


def calculate_intervals(data):
    if len(data) > 0:
        return round(1 + 3.322 * np.log(len(data)))
    return 1  # Минимальное количество Интервалов если данных нет


def histogram(data, k):
    n_bins = np.zeros(k, dtype=int)  # Количество элементов в каждом бине
    x = np.zeros(k + 1)  # Границы бинов
    p = np.zeros(k)  # Вероятности для каждого бина

    delta = (np.max(data) - np.min(data)) / k  # Ширина каждого бина
    x[0] = np.min(data)  # Первая граница

    # Предварительные значения для оптимизации производительности
    data_mean = np.mean(data)
    data_std = np.std(data, ddof=1)

    for m in range(k):
        x[m + 1] = x[m] + delta  # Установка следующей границы
        condition = (
            (data <= x[m + 1]) & (data > x[m]) if m > 0
            else (data <= x[m + 1]) & (data >= x[m])
        )
        n_bins[m] = np.sum(condition)  # Подсчет элементов в текущем бине

        # Вычисление вероятностей для каждого бина
        left_cdf = norm.cdf((x[m] - data_mean) / data_std)
        right_cdf = norm.cdf((x[m + 1] - data_mean) / data_std)
        p[m] = right_cdf - left_cdf

    return x, n_bins, p  # Возвращаем границы, количество и вероятности


def chi_square(n_bins, p, data_len, k, data_name):
    expect = p * data_len
    expect = np.where(expect == 0, 1e-10, expect)
    chi = np.sum((n_bins - expect) ** 2 / expect)

    h = np.arange(1, min(25, k + 1))
    plt.figure(data_name)
    plt.plot(h, p[:24] * data_len, "r", label="P_{теор}")
    plt.bar(h, n_bins[:24], alpha=0.5, label="n(k)")
    plt.grid(True)
    plt.legend()
    plt.xlabel("Интервал")
    plt.ylabel("Частота")
    plt.title("Критерий Хи-квадрат")
    plt.show()

    print("Проверка по критерию Хи-квадрат")
    chi2_threshold = chi2.ppf(0.95, df=k - 3)
    if chi <= chi2_threshold:
        print(f"Гипотеза принята ({chi:.4f} <= {chi2_threshold:.4f})")
    else:
        print(f"Гипотеза не принята ({chi:.4f} > {chi2_threshold:.4f})")


def calculate_empirical_and_theoretical_distribution(data):
    sorted_data = np.sort(data)
    n = len(data)
    F_emp = np.arange(1, n + 1) / n
    mean = np.mean(data)
    std = np.std(data)
    F_theor = norm.cdf(sorted_data, loc=mean, scale=std)

    return sorted_data, F_emp, F_theor


def kolmogorov_test(sorted_data, F_emp, F_theor):
    D = np.max(np.abs(F_emp - F_theor))

    n = len(sorted_data)

    critical_value = 1.36 / np.sqrt(n)

    print(f"Критерий Колмогорова: {'Гипотеза принята' if D <= critical_value else 'Гипотеза не принята'} (D={D:.4f})")


def plot_distribution(sorted_data, F_emp, F_theor):
    plt.step(sorted_data, F_emp, where="post", label="Эмпирическая функция распределения")
    plt.plot(sorted_data, F_theor, linestyle="--", label="Теоретическая функция распределения")
    plt.grid(True)
    plt.xlabel("Значение χ")
    plt.ylabel("Функция распределения")
    plt.legend()
    plt.title("Критерий Колмогорова")
    plt.show()


# корреляция
def autokorr(data):
    n = len(data)
    mean = np.mean(data)
    autocorr = np.correlate(data - mean, data - mean, mode="full") / (n * np.var(data))
    return autocorr[n - 1:]


def plot_autocorrelation(autocorr):
    plt.plot(autocorr, label="Корреляционная функция")
    plt.grid(True)
    plt.xlabel("Задержка τ")
    plt.ylabel("R(τ)")
    plt.title("Корреляционная функция")
    plt.show()


# Проверка стационарности
def split_array_into_segments(data, num_segments=10):
    segment_length = len(data) // num_segments
    return [data[i * segment_length:(i + 1) * segment_length] for i in range(num_segments)]


def check_stationarity_by_variance(segments, kohran_threshold=0.2434):
    variances = [np.var(segment, ddof=1) for segment in segments]
    max_variance = max(variances)
    sum_variance = sum(variances)
    g = max_variance / sum_variance

    print("Проверка стационарности по дисперсии: ")
    if g <= kohran_threshold:
        print(f"Гипотеза принята ({g:.4f} <= {kohran_threshold:.4f})\n")
    else:
        print(f"Гипотеза не принята ({g:.4f} > {kohran_threshold:.4f})\n")


def check_stationarity_by_mean(segments, alpha=0.05):
    _, p_value = f_oneway(*segments)
    print("Проверка стационарности по математическому ожиданию: ")
    if p_value > alpha:
        print(f"Гипотеза принята: p-value={p_value:.4f}")
    else:
        print(f"Гипотеза отклонена: p-value={p_value:.4f}")


# Проверка эргодичности
def plot_normalized_autocorrelation(normalized_autocorr, rq=0.195):
    max_value = np.max(normalized_autocorr)
    print(f"Эргодичность: нормированная автокорреляция максимальное значение={max_value:.4f}")
    plt.plot(np.arange(len(normalized_autocorr)), normalized_autocorr, "r", label="|r_{x}(τ)|")
    plt.axhline(y=rq, color="b", linestyle="-", label=f"rq = {rq}")
    plt.xlabel("Задержка τ")
    plt.ylabel("|r_{x}(τ)|")
    plt.legend()
    plt.title("Проверка эргодичности")
    plt.show()


# Основная функция
def process_signal(data_name, data):
    print(f"--- {data_name} ---")

    time = range(len(data))

    # Построение графика с выбросами
    graph_data(data, time, "До удаления выбросов", f"{data_name}_raw")

    # Удаление выбросов
    cleaned_data = error_detection(data)

    # Построение графика без выбросов
    graph_data(cleaned_data, time[:len(cleaned_data)], "После удаления выбросов", f"{data_name}_cleaned")

    A = skew(cleaned_data)
    M = kurtosis(cleaned_data, fisher=True)
    DA = (6 * (cleaned_data.size - 1)) / (
            (cleaned_data.size + 1) * (cleaned_data.size + 3)
    )
    DE = (
                 24 * cleaned_data.size * (cleaned_data.size - 2) * (cleaned_data.size - 3)
         ) / (
                 (cleaned_data.size + 1) ** 2
                 * (cleaned_data.size + 3)
                 * (cleaned_data.size + 5)
         )

    if abs(A) <= (3 * np.sqrt(DA)):
        print(
            f"Асимметрия: гипотеза принята для {data_name} выборки ({abs(A):.4f} <= {3 * np.sqrt(DA):.4f})"
        )
    else:
        print(
            f"Асимметрия: гипотеза не принята для {data_name} выборки ({abs(A):.4f} > {3 * np.sqrt(DA):.4f})"
        )

    if abs(M) <= (5 * np.sqrt(DE)):
        print(
            f"Эксцесс: гипотеза принята для {data_name} выборки ({abs(M):.4f} <= {5 * np.sqrt(DE):.4f})\n"
        )
    else:
        print(
            f"Эксцесс: гипотеза не принята для {data_name} выборки ({abs(M):.4f} > {5 * np.sqrt(DE):.4f})\n"
        )

    # Гистограмма
    k = calculate_intervals(cleaned_data)
    x, n_bins, p = histogram(cleaned_data, k)
    chi_square(n_bins, p, len(cleaned_data), k, data_name)

    # Критерий Колмогорова
    sorted_data, F_emp, F_theor = calculate_empirical_and_theoretical_distribution(cleaned_data)
    kolmogorov_test(sorted_data, F_emp, F_theor)
    plot_distribution(sorted_data, F_emp, F_theor)

    # Автокорреляция
    R = autokorr(cleaned_data)
    plot_autocorrelation(R)

    # Проверка стационарности
    segments = split_array_into_segments(cleaned_data, num_segments=10)
    check_stationarity_by_variance(segments)
    check_stationarity_by_mean(segments)

    # Проверка эргодичности
    plot_normalized_autocorrelation(np.abs(R / np.mean(cleaned_data)))


def main():
    # Открытие диалогового окна для выбора файла
    root = Tk()
    root.withdraw()
    filename = filedialog.askopenfilename(title="Выберите файл",
                                          filetypes=[("CSV Files", "*.csv"), ("Text Files", "*.txt"),
                                                     ("All Files", "*.*")])

    if not filename:
        print("Файл не выбран.")
        return

    data = pd.read_csv(filename, delimiter="\t").dropna(axis=1, how="all").values.flatten()

    s = len(data) // 6

    A, B, C, D, E = data[:s], data[s:2 * s], data[2 * s:3 * s], data[3 * s:4 * s], data[4 * s:5 * s]

    data_sets = {"A": A, "B": B, "C": C, "D": D, "E": E}

    for name, subset in data_sets.items():
        process_signal(name, subset)


def process_signal(name, subset):
    print(f"Processing {name} with {len(subset)} elements")


if __name__ == "__main__":
    main()