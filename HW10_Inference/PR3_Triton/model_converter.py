#!/usr/bin/env python3

"""
This script converts a PyTorch model to ONNX format for use with Triton Inference Server.
"""

import os
import torch
from torchvision import models
import argparse

def convert_to_onnx(model_name, output_path, batch_size=1):
    """
    Convert a PyTorch model to ONNX format.

    Args:
        model_name: Name of the model to convert (e.g., 'resnet50')
        output_path: Path to save the ONNX model
        batch_size: Batch size for the model
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load pre-trained model
    if model_name.lower() == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif model_name.lower() == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Set model to evaluation mode
    model.eval()

    # Create dummy input tensor
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    # Export the model to ONNX format
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"Model converted and saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch model to ONNX format')
    parser.add_argument('--model', type=str, default='resnet50', help='Model name (resnet50, mobilenet_v2)')
    parser.add_argument('--output', type=str, default='model_repository/resnet50/1/model.onnx', 
                      help='Output path for the ONNX model')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for the model')

    args = parser.parse_args()
    convert_to_onnx(args.model, args.output, args.batch_size)
#!/usr/bin/env python3

"""
Конвертує модель PyTorch в формат ONNX для використання з Triton Inference Server.

Приклад використання:
    python model_converter.py --model resnet50 --output ./model_repository/resnet50/1/model.onnx
"""

import argparse
import os
import torch
import torchvision.models as models

def get_model(model_name):
    """Повертає попередньо натреновану модель за назвою."""
    try:
        if hasattr(models, model_name):
            model_fn = getattr(models, model_name)
            model = model_fn(pretrained=True)
            return model
        else:
            raise ValueError(f"Невідома модель: {model_name}")
    except Exception as e:
        print(f"Помилка завантаження моделі {model_name}: {e}")
        raise

def convert_to_onnx(model, output_path, input_shape=(1, 3, 224, 224)):
    """Конвертує модель PyTorch в формат ONNX."""
    # Створюємо фіктивні вхідні дані для моделі
    dummy_input = torch.randn(input_shape)

    # Переводимо модель в режим виводу
    model.eval()

    # Створюємо директорію для вихідного файлу, якщо вона не існує
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Експортуємо модель в формат ONNX
    torch.onnx.export(
        model,               # модель PyTorch
        dummy_input,        # вхідний тензор
        output_path,        # шлях для збереження
        export_params=True, # зберігати параметри моделі
        opset_version=12,   # версія ONNX
        do_constant_folding=True, # оптимізація констант
        input_names=['input'],    # назви входів
        output_names=['output'],  # назви виходів
        dynamic_axes={'input': {0: 'batch_size'},  # динамічні осі
                      'output': {0: 'batch_size'}}
    )

    print(f"Модель успішно експортована в {output_path}")

def create_config_file(model_name, output_dir, input_shape, output_shape):
    """Створює config.pbtxt для моделі Triton."""
    config_path = os.path.join(output_dir, 'config.pbtxt')

    # Якщо вказаний шлях до конкретного файлу моделі, отримуємо директорію моделі
    if os.path.basename(output_dir).endswith('.onnx'):
        output_dir = os.path.dirname(os.path.dirname(output_dir))
        config_path = os.path.join(output_dir, 'config.pbtxt')

    # Створюємо конфігурацію
    config = f'''name: "{model_name}"
platform: "onnxruntime_onnx"

input [
  {{
    name: "input"
    data_type: TYPE_FP32
    dims: [ {', '.join(map(str, input_shape))} ]
  }}
]

output [
  {{
    name: "output"
    data_type: TYPE_FP32
    dims: [ {', '.join(map(str, output_shape))} ]
  }}
]

instance_group [
  {{
    count: 1
    kind: KIND_CPU
  }}
]
'''

    with open(config_path, 'w') as f:
        f.write(config)

    print(f"Файл конфігурації створено: {config_path}")

def main():
    parser = argparse.ArgumentParser(description='Конвертація моделі PyTorch в ONNX')
    parser.add_argument('--model', type=str, required=True, help='Назва моделі (наприклад, resnet50)')
    parser.add_argument('--output', type=str, required=True, help='Шлях для збереження ONNX файлу')
    parser.add_argument('--input_shape', type=str, default='1,3,224,224', help='Форма вхідних даних (наприклад, 1,3,224,224)')
    parser.add_argument('--create_config', action='store_true', help='Створити config.pbtxt для Triton')

    args = parser.parse_args()

    # Парсимо розміри вхідних даних
    input_shape = tuple(map(int, args.input_shape.split(',')))

    # Завантажуємо модель
    model = get_model(args.model)

    # Визначаємо форму виходу моделі
    dummy_input = torch.randn(input_shape)
    with torch.no_grad():
        output = model(dummy_input)
    output_shape = tuple(output.shape)

    # Конвертуємо модель
    convert_to_onnx(model, args.output, input_shape)

    # Створюємо файл конфігурації, якщо потрібно
    if args.create_config:
        model_name = os.path.basename(os.path.dirname(os.path.dirname(args.output)))
        create_config_file(model_name, os.path.dirname(os.path.dirname(args.output)), 
                          input_shape, output_shape)

if __name__ == '__main__':
    main()
if __name__ == '__main__':
    main()
