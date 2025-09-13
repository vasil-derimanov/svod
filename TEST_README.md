# Video Orientation Detector Testing Scripts

Тези скриптове са създадени за тестване и сравнение на различни версии на детектора за ориентация на видео файлове.

## Налични скриптове

### 1. test_single.py - Тест на единичен видео файл
```bash
python test_single.py video.mp4
```

**Описание:** Изпълнява и двете версии (стара и нова) на един видео файл и сравнява резултатите.

**Параметри:**
- `video` - път до видео файла за тестване
- `--old-script` - път до старата версия (по подразбиране: video_orientation_detector_old.py)
- `--new-script` - път до новата версия (по подразбиране: video_orientation_detector.py)

**Пример:**
```bash
python test_single.py "C:\Videos\test_video.mp4"
```

### 2. test_batch.py - Пакетно тестване на множество видео файлове
```bash
python test_batch.py "C:\Videos"
```

**Описание:** Автоматично намира всички видео файлове в дадена папка и ги тества с двете версии.

**Параметри:**
- `directory` - папка съдържаща видео файлове
- `--old-script` - път до старата версия
- `--new-script` - път до новата версия
- `--max-videos` - максимален брой видео файлове за тестване (по подразбиране: 5)

**Пример:**
```bash
python test_batch.py "C:\Videos" --max-videos 10
```

### 3. test_comparison.py - Детайлно сравнение с експорт на резултати
```bash
python test_comparison.py video1.mp4 video2.mp4 --output results.json
```

**Описание:** Подробно сравнение на няколко видео файла с експорт на резултатите в JSON формат.

**Параметри:**
- `videos` - списък с видео файлове за тестване
- `--old-script` - път до старата версия
- `--new-script` - път до новата версия
- `--output` - файл за експорт на резултатите (JSON формат)

**Пример:**
```bash
python test_comparison.py video1.mp4 video2.mp4 video3.mp4 --output comparison_results.json
```

## Как да използваш скриптовете

1. **Подготовка:**
   - Увери се, че имаш и двете версии на детектора:
     - `video_orientation_detector.py` (нова оптимизирана версия)
     - `video_orientation_detector_old.py` (стара версия)

2. **Бърз тест на един файл:**
   ```bash
   python test_single.py "път\до\видео.mp4"
   ```

3. **Пакетно тестване:**
   ```bash
   python test_batch.py "папка\с\видео"
   ```

4. **Детайлно сравнение:**
   ```bash
   python test_comparison.py video1.mp4 video2.mp4 --output results.json
   ```

## Какво показват резултатите

- ✅ **Orientation match** - дали двете версии дават еднакъв резултат за ориентацията
- ⏱️ **Time difference** - разлика във времето за обработка
- 📊 **Success rate** - процент на успешно обработените файлове
- 📈 **Performance metrics** - средни стойности за време и точност

## Съвети

- Започни с `test_single.py` за бърза проверка на един файл
- Използвай `test_batch.py` за тестване на множество файлове
- Използвай `test_comparison.py` когато искаш да експортираш резултатите
- Ако някоя версия се провали, провери дали всички зависимости са инсталирани

## Отстраняване на проблеми

- **Timeout errors:** Увеличи timeout времето в скрипта ако видеата са много големи
- **Import errors:** Увери се, че всички Python пакети са инсталирани
- **Path errors:** Използвай абсолютни пътища до файловете