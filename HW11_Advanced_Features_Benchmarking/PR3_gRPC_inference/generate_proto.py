# Updated version for PR
#!/usr/bin/env python
"""
РЎРєСЂРёРїС‚ РґР»СЏ РіРµРЅРµСЂР°С†С–С— Python-РєРѕРґСѓ Р· .proto С„Р°Р№Р»С–РІ
"""

import os
import sys
import subprocess
from pathlib import Path

def generate_grpc_code(proto_file, output_dir="."):
    """
    Р“РµРЅРµСЂСѓС” Python-РєРѕРґ Р· .proto С„Р°Р№Р»Сѓ Р·Р° РґРѕРїРѕРјРѕРіРѕСЋ protoc

    РџР°СЂР°РјРµС‚СЂРё:
    -----------
    proto_file: С€Р»СЏС… РґРѕ .proto С„Р°Р№Р»Сѓ
    output_dir: РґРёСЂРµРєС‚РѕСЂС–СЏ РґР»СЏ РІРёС…С–РґРЅРёС… С„Р°Р№Р»С–РІ
    """
    proto_file = Path(proto_file)

    if not proto_file.exists():
        print(f"РџРѕРјРёР»РєР°: С„Р°Р№Р» {proto_file} РЅРµ С–СЃРЅСѓС”")
        return False

    try:
        # РљРѕРјР°РЅРґР° РґР»СЏ РіРµРЅРµСЂР°С†С–С— Python-РєРѕРґСѓ
        cmd = [
            "python", "-m", "grpc_tools.protoc",
            f"--proto_path={proto_file.parent}",
            f"--python_out={output_dir}",
            f"--grpc_python_out={output_dir}",
            str(proto_file)
        ]

        print(f"Р’РёРєРѕРЅР°РЅРЅСЏ РєРѕРјР°РЅРґРё: {' '.join(cmd)}")
        subprocess.check_call(cmd)

        print(f"РЈСЃРїС–С€РЅРѕ Р·РіРµРЅРµСЂРѕРІР°РЅРѕ Python-РєРѕРґ Р· {proto_file}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"РџРѕРјРёР»РєР° РїСЂРё РіРµРЅРµСЂР°С†С–С— РєРѕРґСѓ: {e}")
        return False
    except Exception as e:
        print(f"РќРµРѕС‡С–РєСѓРІР°РЅР° РїРѕРјРёР»РєР°: {e}")
        return False

if __name__ == "__main__":
    # РЁР»СЏС… РґРѕ .proto С„Р°Р№Р»Сѓ Р·Р° Р·Р°РјРѕРІС‡СѓРІР°РЅРЅСЏРј
    default_proto = "proto/inference.proto"

    # РћС‚СЂРёРјР°РЅРЅСЏ С€Р»СЏС…Сѓ Р· Р°СЂРіСѓРјРµРЅС‚С–РІ РєРѕРјР°РЅРґРЅРѕРіРѕ СЂСЏРґРєР°, СЏРєС‰Рѕ РІРєР°Р·Р°РЅРѕ
    proto_file = sys.argv[1] if len(sys.argv) > 1 else default_proto

    success = generate_grpc_code(proto_file)

    if not success:
        sys.exit(1)

