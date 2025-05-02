# ONNX to RKNN Converter

Автоматизированный конвертер ONNX-моделей (в частности, ESRGAN) в формат RKNN для Rockchip NPU (rk3566).

## Назначение

Этот проект решает следующие проблемы:

1. **Ебля с RKNN SDK** - Скрипт использует проверенные рабочие версии всех зависимостей в Docker-контейнере
2. **Баг с dynamic_inputs** - Обходит проблему через генерацию отдельных моделей для каждого разрешения входа
3. **Ручная конвертация** - Автоматизирует создание набора RKNN-моделей с разными разрешениями из одного ONNX-файла

## Использование

### Через Docker (рекомендуется)

```bash
docker run --rm -v $(pwd)/input_models:/workspace/input_models \
                -v $(pwd)/output_models:/workspace/output_models \
                onnx-to-rknn:latest \
                --model_source https://example.com/model.onnx \
                --resolutions 1440x384,1536x512 \
                --input_name input
```

#### Параметры

- `--model_source`: URL или локальный путь (в ./input_models) к ONNX-модели
- `--resolutions`: Список разрешений в формате WxH,WxH,... (default: 1440x384,1536x512)
- `--input_name`: Имя входного тензора (default: input)
- `-v`: Включить подробный вывод

### Сборка Docker образа

```bash
docker build -t onnx-to-rknn .
```

### Через docker-compose (для тестирования)

1. Положите ONNX-модель в ./input_models
2. Настройте параметры в docker-compose.yml
3. Запустите:
```bash
docker-compose up
```

## Структура проекта

```
.
├── Dockerfile              # Образ с RKNN SDK и зависимостями
├── docker-compose.yml      # Конфиг для быстрого запуска
├── scripts/
│   └── convert.py         # Основной скрипт конвертации
├── input_models/          # Входные ONNX модели
└── output_models/         # Сконвертированные RKNN модели
```

## Технические детали

- **Входная модель:** ONNX формат (тестировалось с ESRGAN)
- **Выходной формат:** RKNN для rk3566
- **Квантизация:** w16a16i_dfp без повторной квантизации
- **Имена выходных файлов:** `имя_модели_w16a16i_dfp_ШИРИНАxВЫСОТА.rknn`

## Known Issues

- RKNN SDK имеет баг с `dynamic_inputs`, который приводит к segfault для ESRGAN моделей. Поэтому генерируются отдельные модели для каждого разрешения.
