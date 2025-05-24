#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import json
import re
import random
import time
from urllib.parse import urljoin
import os
import csv
import uuid
import logging
from dotenv import load_dotenv
from openai import OpenAI
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from twocaptcha import TwoCaptcha
import subprocess

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("betaren_scrape.log", encoding='utf-8', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BetarenScraper")

# Загрузка настроек из .env
load_dotenv()

# Настройки
SLEEP_RANGE = (5.0, 10.0)
MAX_RETRIES = int(os.getenv('MAX_RETRIES', 5))
OUTPUT_DIR = os.getenv('DOWNLOAD_DIR', 'downloads')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
TWOCAPTCHA_API_KEY = os.getenv('TWOCAPTCHA_API_KEY', '')
CHROMEDRIVER_PATH = os.getenv('CHROMEDRIVER_PATH', 'D:/crawler_risks/chromedriver.exe')

# Проверка ChromeDriver
try:
    chromedriver_version = subprocess.check_output([CHROMEDRIVER_PATH, '--version']).decode().strip()
    logger.info(f"Версия ChromeDriver: {chromedriver_version}")
except Exception as e:
    logger.error(f"Ошибка проверки ChromeDriver: {e}")
    raise SystemExit("ChromeDriver недоступен. Проверьте путь и установку.")

# Проверка ключей API
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY не задан. Переводы невозможны.")
    raise SystemExit("Укажите OPENAI_API_KEY в .env файле.")
else:
    logger.info("OPENAI_API_KEY настроен успешно.")

if not TWOCAPTCHA_API_KEY:
    logger.warning("TWOCAPTCHA_API_KEY не задан. Обход капчи невозможен.")
else:
    logger.info("TWOCAPTCHA_API_KEY настроен успешно.")

# Инициализация OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Список User-Agent
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.6943.142 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.6943.142 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0'
]

# GUID культур
crop_guids = {
    "пшеница": "501933b1-b43d-11e9-a3c5-00155d012200",
    "ячмень": "ed8d52db-9b99-4ac3-9691-082829953c46",
    "кукуруза": "c61ecabe-229e-4361-921a-081125d5c3c1",
    "овес": "c593d0c4-653c-418b-9ac4-35d58f3ff627",
    "рапс": "f503750b-f20d-4bf7-af68-ce922974018c",
    "горчица": "4f091a7c-30f4-4825-a76a-1ffffb89483b",
    "соя": "soy_guid_placeholder",
    "сахарная свекла": "sugar_beet_guid_placeholder",
    "подсолнечник": "sunflower_guid_placeholder",
    "горох": "pea_guid_placeholder",
    "нут": "nut_guid_placeholder",
    "лен": "flax_guid_placeholder",
    "картофель": "potato_guid_placeholder",
    "садовые культуры": "orchard_guid_placeholder",
    "виноградники": "vineyard_guid_placeholder"
}

# Создание директорий
os.makedirs(OUTPUT_DIR, exist_ok=True)
for folder in ["diseases", "pests", "weeds"]:
    os.makedirs(os.path.join(OUTPUT_DIR, "images", folder), exist_ok=True)

def translate_text(text, target_lang):
    if not text or not client:
        logger.warning(f"Пустой текст или клиент OpenAI не инициализирован для перевода на {target_lang}")
        return ''
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Ты профессиональный переводчик, специализирующийся на сельскохозяйственной терминологии."},
                {"role": "user", "content": f"Переведи следующий текст с русского на {target_lang}, сохраняя агрономическую терминологию:\n\n{text}"}
            ],
            max_tokens=2000,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Ошибка перевода текста: {e}")
        return ''

def solve_recaptcha(site_key, url):
    if not TWOCAPTCHA_API_KEY:
        logger.warning("TWOCAPTCHA_API_KEY не задан, пропускаем решение капчи.")
        return None
    solver = TwoCaptcha(TWOCAPTCHA_API_KEY)
    try:
        result = solver.recaptcha(sitekey=site_key, url=url)
        logger.info(f"reCAPTCHA решена: {result['code']}")
        return result['code']
    except Exception as e:
        logger.error(f"Ошибка при решении reCAPTCHA: {e}")
        return None

def fetch_page_content(url, retries=MAX_RETRIES):
    options = Options()
    user_agent = random.choice(USER_AGENTS)
    options.add_argument(f'user-agent={user_agent}')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-geolocation')
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')

    service = Service(CHROMEDRIVER_PATH)
    driver = None

    for attempt in range(retries):
        try:
            driver = webdriver.Chrome(service=service, options=options)
            driver.set_page_load_timeout(30)
            driver.get(url)
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            # Проверка reCAPTCHA
            if 'g-recaptcha' in driver.page_source:
                logger.debug("Обнаружена reCAPTCHA, пытаемся решить...")
                try:
                    site_key = driver.find_element(By.CLASS_NAME, 'g-recaptcha').get_attribute('data-sitekey')
                    captcha_response = solve_recaptcha(site_key, url)
                    if captcha_response:
                        driver.execute_script(f'document.getElementById("g-recaptcha-response").innerHTML="{captcha_response}";')
                        driver.find_element(By.ID, 'feedback-form').submit()
                        WebDriverWait(driver, 15).until(
                            EC.presence_of_element_located((By.TAG_NAME, "body"))
                        )
                        logger.debug("reCAPTCHA решена")
                    else:
                        logger.warning("Не удалось решить reCAPTCHA")
                except Exception as e:
                    logger.error(f"Ошибка при решении reCAPTCHA: {e}")

            html_content = driver.page_source
            if "Не удается получить доступ к сайту" in html_content:
                logger.warning(f"Страница {url} недоступна (ошибка доступа). Повторная попытка ({attempt + 1}/{retries})...")
                driver.quit()
                time.sleep(random.uniform(*SLEEP_RANGE))
                continue

            debug_file = f'debug_page_{uuid.uuid4()}.html'
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.debug(f"HTML сохранен в {debug_file}")
            return html_content
        except Exception as e:
            logger.error(f"Ошибка при загрузке страницы {url} (попытка {attempt + 1}/{retries}): {e}")
            if driver:
                driver.quit()
            time.sleep(random.uniform(*SLEEP_RANGE))
        finally:
            if driver:
                driver.quit()
    
    logger.error(f"Не удалось загрузить страницу {url} после {retries} попыток.")
    return None

def download_image(url, folder, filename, referer):
    if not url or url.startswith('data:image'):
        logger.debug(f"Некорректный URL изображения: {url}")
        return ''
    
    # Создаем полный путь для сохранения
    os.makedirs(os.path.join(OUTPUT_DIR, folder), exist_ok=True)
    image_path = os.path.join(OUTPUT_DIR, folder, filename)
    
    try:
        headers = {
            'User-Agent': random.choice(USER_AGENTS),
            'Referer': referer,
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9,ru;q=0.8',
            'Connection': 'keep-alive'
        }
        
        # Попробуем сначала через requests
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            with open(image_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"Изображение сохранено через requests: {image_path}")
            return image_path
        except Exception as e:
            logger.warning(f"Не удалось скачать изображение через requests: {e}")
            
            # Попробуем через Selenium
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')
            options.add_argument(f'user-agent={random.choice(USER_AGENTS)}')
            
            service = Service(CHROMEDRIVER_PATH)
            driver = webdriver.Chrome(service=service, options=options)
            try:
                driver.get(url)
                time.sleep(2)  # Даем время для загрузки изображения
                
                # Сохраняем скриншот
                driver.save_screenshot(image_path)
                logger.info(f"Изображение сохранено через Selenium: {image_path}")
                return image_path
            except Exception as e:
                logger.error(f"Не удалось скачать изображение через Selenium: {e}")
                return ''
            finally:
                driver.quit()
    except Exception as e:
        logger.error(f"Ошибка при скачивании изображения {url}: {e}")
        return ''

def parse_diseases(url, crop):
    html_content = fetch_page_content(url)
    if not html_content:
        logger.debug(f"Не удалось загрузить контент для {url}")
        return []

    soup = BeautifulSoup(html_content, 'html.parser')
    diseases = []
    
    # Определяем, это страница списка или детальная страница
    is_list_page = bool(soup.find('div', class_='harmful-list'))
    
    if is_list_page:
        # Извлечение списка ссылок на детальные страницы
        links = soup.find_all('a', class_='harmful-list__title')
        detail_urls = []
        for link in links:
            if link.has_attr('href') and '/harmful/bolezni/' in link['href']:
                detail_url = urljoin(url, link['href'])
                if detail_url not in detail_urls:
                    detail_urls.append(detail_url)
        
        # Если ссылок не найдено, попробуем другие селекторы
        if not detail_urls:
            links = soup.find_all('a', href=True)
            for link in links:
                if '/harmful/bolezni/' in link['href'] and 'harmful-list__title' in link.get('class', []):
                    detail_url = urljoin(url, link['href'])
                    if detail_url not in detail_urls:
                        detail_urls.append(detail_url)
        logger.debug(f"Найдено {len(detail_urls)} детальных страниц болезней для {url}")
    else:
        # Это уже детальная страница
        detail_urls = [url]

    for detail_url in detail_urls:
        detail_html = fetch_page_content(detail_url)
        if not detail_html:
            logger.debug(f"Не удалось загрузить детальную страницу {detail_url}")
            continue

        detail_soup = BeautifulSoup(detail_html, 'html.parser')
        
        # Поиск основной информации
        title_elem = detail_soup.find('h1')
        
        # Поиск изображения
        image_elem = None
        gallery_div = detail_soup.find('div', class_='harmful-detail__picture')
        if gallery_div:
            image_elem = gallery_div.find('img', class_='gallery__img')
        
        # Если не нашли картинку в главном блоке, ищем в других местах
        if not image_elem:
            image_elem = detail_soup.find('img', class_='gallery__img')
        
        # Поиск текстовых блоков болезни
        description_blocks = detail_soup.find_all('div', class_='harmful-detail__text')
        
        if not title_elem:
            logger.warning(f"Заголовок не найден на странице {detail_url}, пропускаем")
            continue

        name = title_elem.get_text(strip=True)
        
        # Извлечение научного названия
        scientific_name = ""
        first_text_block = detail_soup.find('div', class_='harmful-detail__text')
        if first_text_block:
            scientific_name_match = re.search(r'Возбудитель -\s*(.*?)(?:\<br\>|\<\/span\>)', str(first_text_block))
            if scientific_name_match:
                scientific_name = scientific_name_match.group(1).strip()
                scientific_name = re.sub(r'<[^>]+>', '', scientific_name)  # Удаление HTML-тегов
        
        # Объединение текстовых блоков или выделение разных частей информации
        description = ""
        symptoms = ""
        development_conditions = ""
        control_measures = ""
        
        if description_blocks and len(description_blocks) > 0:
            # Первый блок обычно содержит общую информацию и научное название
            if len(description_blocks) >= 1:
                description = description_blocks[0].get_text(strip=True)
            
            # Второй блок обычно содержит более подробную информацию
            if len(description_blocks) >= 2:
                full_text = description_blocks[1].get_text(strip=True)
                
                # Поиск симптомов
                symptoms_match = re.search(r'(Симптомы болезни.*?)(?:Факторы|Меры|$)', full_text, re.DOTALL)
                if symptoms_match:
                    symptoms = symptoms_match.group(1).strip()
                
                # Поиск факторов, способствующих развитию
                factors_match = re.search(r'(Факторы, содействующие развитию болезни.*?)(?:Меры|$)', full_text, re.DOTALL)
                if factors_match:
                    development_conditions = factors_match.group(1).strip()
                
                # Поиск мер защиты
                measures_match = re.search(r'(Меры защиты.*?)$', full_text, re.DOTALL)
                if measures_match:
                    control_measures = measures_match.group(1).strip()
        
        # Если разделение по разделам не удалось, используем весь текст второго блока (если есть)
        if len(description_blocks) >= 2 and not (symptoms or development_conditions or control_measures):
            full_text = description_blocks[1].get_text(strip=True)
            # Разделяем текст по заголовкам
            sections = re.split(r'(?:Симптомы болезни|Факторы, содействующие развитию болезни|Меры защиты)', full_text)
            if len(sections) >= 2:
                symptoms = sections[1].strip() if len(sections) > 1 else ""
            if len(sections) >= 3:
                development_conditions = sections[2].strip() if len(sections) > 2 else ""
            if len(sections) >= 4:
                control_measures = sections[3].strip() if len(sections) > 3 else ""
        
        # Если изображение найдено, получаем его URL
        image_url = None
        if image_elem and image_elem.has_attr('src'):
            image_url = urljoin(detail_url, image_elem['src'])
        
        # Проверка, что название не является служебным элементом
        if name.lower() in ['не удается получить доступ к сайту', 'не пропустите', 'подписка', 'обратная связь']:
            logger.debug(f"Пропущен некорректный заголовок: {name}")
            continue
        
        # Генерация ID и сохранение изображения
        disease_id = str(uuid.uuid4())
        image_filename = f"disease_{name.replace(' ', '_').replace('/', '_')}_{disease_id}.jpg"
        image_path = download_image(image_url, 'images/diseases', image_filename, detail_url) if image_url else ''
        
        # Формирование записи о болезни
        disease_data = {
            'id': disease_id,
            'name': name,
            'name_en': translate_text(name, 'en') if name else '',
            'scientific_name': scientific_name,
            'is_active': True,
            'description_ru': description,
            'description_ua': translate_text(description, 'ua') if description else '',
            'description_en': translate_text(description, 'en') if description else '',
            'symptoms_ru': symptoms,
            'symptoms_ua': translate_text(symptoms, 'ua') if symptoms else '',
            'symptoms_en': translate_text(symptoms, 'en') if symptoms else '',
            'development_conditions_ru': development_conditions,
            'development_conditions_ua': translate_text(development_conditions, 'ua') if development_conditions else '',
            'development_conditions_en': translate_text(development_conditions, 'en') if development_conditions else '',
            'control_measures_ru': control_measures,
            'control_measures_ua': translate_text(control_measures, 'ua') if control_measures else '',
            'control_measures_en': translate_text(control_measures, 'en') if control_measures else '',
            'photo_path': image_path,
            'source_urls': detail_url,
            'crops': crop
        }
        
        # Сохранение HTML для отладки
        with open(f'debug_disease_{disease_id}.html', 'w', encoding='utf-8') as f:
            f.write(detail_soup.prettify())
        
        diseases.append(disease_data)
        logger.info(f"Добавлена болезнь: {name} для {crop}")

    return diseases
    
def parse_pests(url, crop):
    html_content = fetch_page_content(url)
    if not html_content:
        logger.debug(f"Не удалось загрузить контент для {url}")
        return []

    soup = BeautifulSoup(html_content, 'html.parser')
    pests = []
    
    sections = soup.find_all(['div', 'section', 'article', 'li', 'a'], class_=re.compile('harmful|pest|item|block|entry|list|link|card'))
    detail_urls = []
    for section in sections:
        link_elem = section.find('a', href=True)
        if link_elem and '/harmful/vrediteli/' in link_elem['href'] and link_elem['href'] != url:
            detail_url = urljoin(url, link_elem['href'])
            if detail_url not in detail_urls:
                detail_urls.append(detail_url)
    
    logger.debug(f"Найдено {len(detail_urls)} детальных страниц вредителей для {url}: {detail_urls}")

    for detail_url in detail_urls:
        detail_html = fetch_page_content(detail_url)
        if not detail_html:
            logger.debug(f"Не удалось загрузить детальную страницу {detail_url}")
            continue

        detail_soup = BeautifulSoup(detail_html, 'html.parser')
        
        title_elem = detail_soup.find(['h1', 'h2'], class_=re.compile('harmful__title|title|name|heading')) or detail_soup.find(['h1', 'h2'])
        description_elem = detail_soup.find(['div', 'p'], class_=re.compile('harmful__content|content|description|info|summary|body')) or detail_soup.find('p')
        damage_elem = detail_soup.find(['div', 'p'], class_=re.compile('damage|symptoms|impact')) or detail_soup.find('p', string=re.compile('ущерб|повреждения', re.I))
        biology_elem = detail_soup.find(['div', 'p'], class_=re.compile('biology|life_cycle|development')) or detail_soup.find('p', string=re.compile('биология|жизненный цикл', re.I))
        measures_elem = detail_soup.find(['div', 'p'], class_=re.compile('control|measures|protection')) or detail_soup.find('p', string=re.compile('меры|борьба|защита', re.I))
        image_elem = detail_soup.find('img', class_=re.compile('harmful__image|image|photo')) or detail_soup.find('img', src=True)

        logger.debug(f"Детальная страница {detail_url}: Заголовок={title_elem is not None}, Описание={description_elem is not None}, Ущерб={damage_elem is not None}, Биология={biology_elem is not None}, Меры={measures_elem is not None}, Изображение={image_elem is not None}")

        if not title_elem:
            logger.warning(f"Заголовок не найден на странице {detail_url}, пропускаем")
            continue

        name = title_elem.get_text(strip=True)
        description = description_elem.get_text(strip=True) if description_elem else ''
        damage = damage_elem.get_text(strip=True) if damage_elem else description
        biology = biology_elem.get_text(strip=True) if biology_elem else ''
        measures = measures_elem.get_text(strip=True) if measures_elem else ''
        image_url = urljoin(detail_url, image_elem['src']) if image_elem and 'company-logo' not in image_elem.get('src', '') else None

        if name.lower() in ['не удается получить доступ к сайту', 'не пропустите', 'подписка', 'обратная связь']:
            logger.debug(f"Пропущен некорректный заголовок: {name}")
            continue

        pest_id = str(uuid.uuid4())
        image_filename = f"pest_{name.replace(' ', '_').replace('/', '_')}_{pest_id}.jpg"
        image_path = download_image(image_url, 'images/pests', image_filename, detail_url) if image_url else ''

        pests.append({
            'id': pest_id,
            'name': name,
            'name_en': translate_text(name, 'en') if name else '',
            'scientific_name': '',
            'is_active': True,
            'description_ru': description,
            'description_ua': translate_text(description, 'ua') if description else '',
            'description_en': translate_text(description, 'en') if description else '',
            'damage_symptoms_ru': damage,
            'damage_symptoms_ua': translate_text(damage, 'ua') if damage else '',
            'damage_symptoms_en': translate_text(damage, 'en') if damage else '',
            'biology_ru': biology,
            'biology_ua': translate_text(biology, 'ua') if biology else '',
            'biology_en': translate_text(biology, 'en') if biology else '',
            'control_measures_ru': measures,
            'control_measures_ua': translate_text(measures, 'ua') if measures else '',
            'control_measures_en': translate_text(measures, 'en') if measures else '',
            'photo_path': image_path,
            'source_urls': detail_url,
            'crops': crop
        })
        logger.info(f"Добавлен вредитель: {name} для {crop}")

        with open(f'debug_pest_{pest_id}.html', 'w', encoding='utf-8') as f:
            f.write(detail_soup.prettify())
        logger.debug(f"HTML детальной страницы сохранен в debug_pest_{pest_id}.html")

    logger.debug(f"Итоговый список вредителей: {len(pests)} записей")
    return pests

def parse_weeds(url):
    html_content = fetch_page_content(url)
    if not html_content:
        logger.debug(f"Не удалось загрузить контент для {url}")
        return []

    soup = BeautifulSoup(html_content, 'html.parser')
    weeds = []
    
    sections = soup.find_all(['div', 'section', 'article', 'li', 'a'], class_=re.compile('harmful|weed|item|block|entry|list|link|card'))
    detail_urls = []
    for section in sections:
        link_elem = section.find('a', href=True)
        if link_elem and '/harmful/sornyaki/' in link_elem['href'] and link_elem['href'] != url:
            detail_url = urljoin(url, link_elem['href'])
            if detail_url not in detail_urls:
                detail_urls.append(detail_url)
    
    logger.debug(f"Найдено {len(detail_urls)} детальных страниц сорняков для {url}: {detail_urls}")

    for detail_url in detail_urls:
        detail_html = fetch_page_content(detail_url)
        if not detail_html:
            logger.debug(f"Не удалось загрузить детальную страницу {detail_url}")
            continue

        detail_soup = BeautifulSoup(detail_html, 'html.parser')
        
        title_elem = detail_soup.find(['h1', 'h2'], class_=re.compile('harmful__title|title|name|heading')) or detail_soup.find(['h1', 'h2'])
        scientific_elem = detail_soup.find(['p', 'span', 'i'], class_=re.compile('scientific|latin|name')) or detail_soup.find('i')
        description_elem = detail_soup.find(['div', 'p'], class_=re.compile('harmful__content|content|description|info|summary|body')) or detail_soup.find('p')
        features_elem = detail_soup.find(['div', 'p'], class_=re.compile('features|biology|characteristics')) or detail_soup.find('p', string=re.compile('биология|характеристики', re.I))
        harmfulness_elem = detail_soup.find(['div', 'p'], class_=re.compile('harmfulness|damage|impact')) or detail_soup.find('p', string=re.compile('вредоносность|ущерб', re.I))
        measures_elem = detail_soup.find(['div', 'p'], class_=re.compile('control|measures|protection')) or detail_soup.find('p', string=re.compile('меры|борьба|защита', re.I))
        image_elem = detail_soup.find('img', class_=re.compile('harmful__image|image|photo')) or detail_soup.find('img', src=True)

        logger.debug(f"Детальная страница {detail_url}: Заголовок={title_elem is not None}, Научное имя={scientific_elem is not None}, Описание={description_elem is not None}, Биология={features_elem is not None}, Вредоносность={harmfulness_elem is not None}, Меры={measures_elem is not None}, Изображение={image_elem is not None}")

        if not title_elem:
            logger.warning(f"Заголовок не найден на странице {detail_url}, пропускаем")
            continue

        name = title_elem.get_text(strip=True)
        scientific_name = scientific_elem.get_text(strip=True) if scientific_elem else ''
        description = description_elem.get_text(strip=True) if description_elem else ''
        features = features_elem.get_text(strip=True) if features_elem else description
        harmfulness = harmfulness_elem.get_text(strip=True) if harmfulness_elem else ''
        measures = measures_elem.get_text(strip=True) if measures_elem else ''
        image_url = urljoin(detail_url, image_elem['src']) if image_elem and 'company-logo' not in image_elem.get('src', '') else None

        if name.lower() in ['не удается получить доступ к сайту', 'не пропустите', 'подписка', 'обратная связь']:
            logger.debug(f"Пропущен некорректный заголовок: {name}")
            continue

        weed_id = str(uuid.uuid4())
        image_filename = f"weed_{name.replace(' ', '_').replace('/', '_')}_{weed_id}.jpg"
        image_path = download_image(image_url, 'images/weeds', image_filename, detail_url) if image_url else ''

        weeds.append({
            'id': weed_id,
            'name': name,
            'name_en': translate_text(name, 'en') if name else '',
            'scientific_name': scientific_name,
            'is_active': True,
            'description_ru': description,
            'description_ua': translate_text(description, 'ua') if description else '',
            'description_en': translate_text(description, 'en') if description else '',
            'biological_features_ru': features,
            'biological_features_ua': translate_text(features, 'ua') if features else '',
            'biological_features_en': translate_text(features, 'en') if features else '',
            'harmfulness_ru': harmfulness,
            'harmfulness_ua': translate_text(harmfulness, 'ua') if harmfulness else '',
            'harmfulness_en': translate_text(harmfulness, 'en') if harmfulness else '',
            'control_measures_ru': measures,
            'control_measures_ua': translate_text(measures, 'ua') if measures else '',
            'control_measures_en': translate_text(measures, 'en') if measures else '',
            'photo_path': image_path,
            'source_urls': detail_url
        })
        logger.info(f"Добавлен сорняк: {name}")

        with open(f'debug_weed_{weed_id}.html', 'w', encoding='utf-8') as f:
            f.write(detail_soup.prettify())
        logger.debug(f"HTML детальной страницы сохранен в debug_weed_{weed_id}.html")

    logger.debug(f"Итоговый список сорняков: {len(weeds)} записей")
    return weeds

def save_to_csv(data, filename, fieldnames):
    if not data:
        logger.warning(f"Нет данных для сохранения в {filename}")
        return False
    
    # Создаем директорию, если её нет
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    try:
        # Преобразуем данные в строковый формат, если нужно
        processed_data = []
        for row in data:
            processed_row = {}
            for k, v in row.items():
                if k in fieldnames:
                    if v is None:
                        processed_row[k] = ''
                    else:
                        processed_row[k] = str(v)
            processed_data.append(processed_row)
        
        # Записываем в CSV
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in processed_data:
                writer.writerow(row)
        
        logger.info(f"Сохранено {len(data)} записей в {filename}")
        return True
    except Exception as e:
        logger.error(f"Ошибка при сохранении CSV {filename}: {e}")
        
        # Попробуем сохранить с другой кодировкой
        try:
            with open(filename, 'w', newline='', encoding='cp1251') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in processed_data:
                    writer.writerow(row)
            logger.info(f"Сохранено {len(data)} записей в {filename} с кодировкой cp1251")
            return True
        except Exception as e2:
            logger.error(f"Вторая попытка сохранения CSV {filename} не удалась: {e2}")
            return False

def check_environment():
    """Проверяет наличие необходимых директорий и прав на запись."""
    logger.info("Проверка среды выполнения...")
    
    # Проверка существования OUTPUT_DIR
    if not os.path.exists(OUTPUT_DIR):
        try:
            os.makedirs(OUTPUT_DIR)
            logger.info(f"Создана директория {OUTPUT_DIR}")
        except Exception as e:
            logger.error(f"Не удалось создать директорию {OUTPUT_DIR}: {e}")
            return False
    
    # Проверка прав на запись в OUTPUT_DIR
    test_file = os.path.join(OUTPUT_DIR, "test_write.txt")
    try:
        with open(test_file, 'w') as f:
            f.write("Test")
        os.remove(test_file)
        logger.info(f"Есть права на запись в {OUTPUT_DIR}")
    except Exception as e:
        logger.error(f"Нет прав на запись в {OUTPUT_DIR}: {e}")
        return False
    
    # Проверка директорий для изображений
    for folder in ["images", "images/diseases", "images/pests", "images/weeds"]:
        path = os.path.join(OUTPUT_DIR, folder)
        if not os.path.exists(path):
            try:
                os.makedirs(path)
                logger.info(f"Создана директория {path}")
            except Exception as e:
                logger.error(f"Не удалось создать директорию {path}: {e}")
                return False
    
    return True
    
def main():
    # Проверка среды выполнения
    if not check_environment():
        logger.error("Проверка среды выполнения не пройдена. Завершение работы.")
        return
    
    base_url = 'https://betaren.ru'
    urls = {
        'diseases': {
            'cereals': f'{base_url}/harmful/bolezni/bolezni-zernovykh-kultur/',
            'rapeseed': f'{base_url}/harmful/bolezni/bolezni-rapsa/',
            'corn': f'{base_url}/harmful/bolezni/bolezni-kukuruzy/',
            'soy': f'{base_url}/harmful/bolezni/bolezni-soi/',
            'sugar_beet': f'{base_url}/harmful/bolezni/bolezni-sakharnoy-svekly/',
            'sunflower': f'{base_url}/harmful/bolezni/bolezni-podsolnechnika/',
            'pea_nut': f'{base_url}/harmful/bolezni/bolezni-gorokha-i-nuta/',
            'flax': f'{base_url}/harmful/bolezni/bolezni-lna/',
            'potato': f'{base_url}/harmful/bolezni/bolezni-kartofelya/',
            'orchard': f'{base_url}/harmful/bolezni/bolezni-sadovykh-kultur/',
            'vineyard': f'{base_url}/harmful/bolezni/bolezni-vinogradnikov/'
        },
        'pests': {
            'cereals': f'{base_url}/harmful/vrediteli/vrediteli-zernovykh-kultur/',
            'rapeseed': f'{base_url}/harmful/vrediteli/vrediteli-rapsa/',
            'corn': f'{base_url}/harmful/vrediteli/vrediteli-kukuruzy/',
            'soy': f'{base_url}/harmful/vrediteli/vrediteli-soi/',
            'sugar_beet': f'{base_url}/harmful/vrediteli/vrediteli-sakharnoy-svekly/',
            'sunflower': f'{base_url}/harmful/vrediteli/vrediteli-podsolnechnika/',
            'pea_nut': f'{base_url}/harmful/vrediteli/vrediteli-gorokha-i-nuta/',
            'flax': f'{base_url}/harmful/vrediteli/vrediteli-lna/',
            'potato': f'{base_url}/harmful/vrediteli/vrediteli-kartofelya/',
            'orchard': f'{base_url}/harmful/vrediteli/vrediteli-sadovykh-kultur/',
            'vineyard': f'{base_url}/harmful/vrediteli/vrediteli-vinogradnikov/'
        },
        'weeds': f'{base_url}/harmful/sornyaki/'
    }

    # Для отладки одной страницы
    debug_specific_page = os.getenv('DEBUG_SPECIFIC_PAGE', '')
    if debug_specific_page:
        logger.info(f"Запуск в режиме отладки для страницы: {debug_specific_page}")
        test_diseases = parse_diseases(debug_specific_page, "пшеница")
        
        if test_diseases:
            logger.info(f"Успешно получено {len(test_diseases)} болезней")
            
            # Сохраняем для отладки в JSON
            with open(os.path.join(OUTPUT_DIR, 'debug_diseases.json'), 'w', encoding='utf-8') as f:
                json.dump(test_diseases, f, ensure_ascii=False, indent=4)
            
            # Структурируем данные для CSV
            diseases = []
            disease_descriptions = []
            disease_crops = []
            disease_images = []
            
            for disease in test_diseases:
                disease_id = disease['id']
                desc_id = str(uuid.uuid4())
                img_id = str(uuid.uuid4())
                
                diseases.append({
                    'id': disease_id,
                    'name': disease['name'],
                    'name_en': disease['name_en'],
                    'scientific_name': disease['scientific_name'],
                    'is_active': disease['is_active']
                })
                
                disease_descriptions.append({
                    'id': desc_id,
                    'disease_id': disease_id,
                    'description_ru': disease['description_ru'],
                    'description_ua': disease['description_ua'],
                    'description_en': disease['description_en'],
                    'symptoms_ru': disease['symptoms_ru'],
                    'symptoms_ua': disease['symptoms_ua'],
                    'symptoms_en': disease['symptoms_en'],
                    'development_conditions_ru': disease['development_conditions_ru'],
                    'development_conditions_ua': disease['development_conditions_ua'],
                    'development_conditions_en': disease['development_conditions_en'],
                    'control_measures_ru': disease['control_measures_ru'],
                    'control_measures_ua': disease['control_measures_ua'],
                    'control_measures_en': disease['control_measures_en'],
                    'photo_path': disease['photo_path'],
                    'source_urls': disease['source_urls'],
                    'version': 1
                })
                
                disease_crops.append({
                    'disease_id': disease_id,
                    'crops': disease['crops']
                })
                
                if disease['photo_path']:
                    disease_images.append({
                        'id': img_id,
                        'disease_id': disease_id,
                        'image_url': disease['source_urls'],
                        'image_path': disease['photo_path'],
                        'version': 1
                    })
            
            # Сохраняем CSV
            save_to_csv(diseases, os.path.join(OUTPUT_DIR, 'debug_diseases.csv'),
                        ['id', 'name', 'name_en', 'scientific_name', 'is_active'])
            save_to_csv(disease_descriptions, os.path.join(OUTPUT_DIR, 'debug_disease_descriptions.csv'),
                        ['id', 'disease_id', 'description_ru', 'description_ua', 'description_en',
                         'symptoms_ru', 'symptoms_ua', 'symptoms_en',
                         'development_conditions_ru', 'development_conditions_ua', 'development_conditions_en',
                         'control_measures_ru', 'control_measures_ua', 'control_measures_en',
                         'photo_path', 'source_urls', 'version'])
            save_to_csv(disease_crops, os.path.join(OUTPUT_DIR, 'debug_disease_crops.csv'),
                        ['disease_id', 'crops'])
            save_to_csv(disease_images, os.path.join(OUTPUT_DIR, 'debug_disease_images.csv'),
                        ['id', 'disease_id', 'image_url', 'image_path', 'version'])
        else:
            logger.error("Не удалось получить информацию о болезнях в режиме отладки")
        
        return
    
    # Инициализация списков для хранения данных
    diseases = []
    disease_descriptions = []
    disease_crops = []
    disease_images = []
    pests = []
    pest_descriptions = []
    pest_crops = []
    pest_images = []
    weeds = []
    weed_descriptions = []
    weed_crops = []
    weed_images = []

    # Соответствие между кодом категории и списком культур
    crop_mapping = {
        'cereals': ['пшеница', 'ячмень', 'овес'],
        'rapeseed': ['рапс'],
        'corn': ['кукуруза'],
        'soy': ['соя'],
        'sugar_beet': ['сахарная свекла'],
        'sunflower': ['подсолнечник'],
        'pea_nut': ['горох', 'нут'],
        'flax': ['лен'],
        'potato': ['картофель'],
        'orchard': ['садовые культуры'],
        'vineyard': ['виноградники']
    }

    # Парсинг болезней для каждой культуры
    for category, crop_url in urls['diseases'].items():
        for crop_name in crop_mapping[category]:
            logger.info(f"Парсинг болезней для культуры: {crop_name}")
            try:
                crop_diseases = parse_diseases(crop_url, crop_name)
                logger.debug(f"Извлечено {len(crop_diseases)} болезней для {crop_name}: {[d['name'] for d in crop_diseases]}")
                
                for disease in crop_diseases:
                    disease_id = disease['id']
                    desc_id = str(uuid.uuid4())
                    img_id = str(uuid.uuid4())

                    diseases.append({
                        'id': disease_id,
                        'name': disease['name'],
                        'name_en': disease['name_en'],
                        'scientific_name': disease['scientific_name'],
                        'is_active': disease['is_active']
                    })

                    disease_descriptions.append({
                        'id': desc_id,
                        'disease_id': disease_id,
                        'description_ru': disease['description_ru'],
                        'description_ua': disease['description_ua'],
                        'description_en': disease['description_en'],
                        'symptoms_ru': disease['symptoms_ru'],
                        'symptoms_ua': disease['symptoms_ua'],
                        'symptoms_en': disease['symptoms_en'],
                        'development_conditions_ru': disease['development_conditions_ru'],
                        'development_conditions_ua': disease['development_conditions_ua'],
                        'development_conditions_en': disease['development_conditions_en'],
                        'control_measures_ru': disease['control_measures_ru'],
                        'control_measures_ua': disease['control_measures_ua'],
                        'control_measures_en': disease['control_measures_en'],
                        'photo_path': disease['photo_path'],
                        'source_urls': disease['source_urls'],
                        'version': 1
                    })

                    disease_crops.append({
                        'disease_id': disease_id,
                        'crops': crop_name
                    })

                    if disease['photo_path']:
                        disease_images.append({
                            'id': img_id,
                            'disease_id': disease_id,
                            'image_url': disease['source_urls'],
                            'image_path': disease['photo_path'],
                            'version': 1
                        })
                
                # Сохраняем промежуточные результаты после каждой культуры
                if crop_diseases:
                    temp_filename = f'diseases_{crop_name}_{category}.json'
                    with open(os.path.join(OUTPUT_DIR, temp_filename), 'w', encoding='utf-8') as f:
                        json.dump(crop_diseases, f, ensure_ascii=False, indent=4)
                    logger.info(f"Сохранен промежуточный файл {temp_filename}")
            except Exception as e:
                logger.error(f"Ошибка при парсинге болезней для {crop_name}: {e}")
                # Продолжаем с другими культурами

    # Парсинг вредителей для каждой культуры
    for category, crop_url in urls['pests'].items():
        for crop_name in crop_mapping[category]:
            logger.info(f"Парсинг вредителей для культуры: {crop_name}")
            try:
                crop_pests = parse_pests(crop_url, crop_name)
                logger.debug(f"Извлечено {len(crop_pests)} вредителей для {crop_name}: {[p['name'] for p in crop_pests]}")
                
                for pest in crop_pests:
                    pest_id = pest['id']
                    desc_id = str(uuid.uuid4())
                    img_id = str(uuid.uuid4())

                    pests.append({
                        'id': pest_id,
                        'name': pest['name'],
                        'name_en': pest['name_en'],
                        'scientific_name': pest['scientific_name'],
                        'is_active': pest['is_active']
                    })

                    pest_descriptions.append({
                        'id': desc_id,
                        'vermin_id': pest_id,
                        'description_ru': pest['description_ru'],
                        'description_ua': pest['description_ua'],
                        'description_en': pest['description_en'],
                        'damage_symptoms_ru': pest['damage_symptoms_ru'],
                        'damage_symptoms_ua': pest['damage_symptoms_ua'],
                        'damage_symptoms_en': pest['damage_symptoms_en'],
                        'biology_ru': pest['biology_ru'],
                        'biology_ua': pest['biology_ua'],
                        'biology_en': pest['biology_en'],
                        'control_measures_ru': pest['control_measures_ru'],
                        'control_measures_ua': pest['control_measures_ua'],
                        'control_measures_en': pest['control_measures_en'],
                        'photo_path': pest['photo_path'],
                        'source_urls': pest['source_urls'],
                        'version': 1
                    })

                    pest_crops.append({
                        'vermin_id': pest_id,
                        'crops': crop_name
                    })

                    if pest['photo_path']:
                        pest_images.append({
                            'id': img_id,
                            'vermin_id': pest_id,
                            'image_url': pest['source_urls'],
                            'image_path': pest['photo_path'],
                            'version': 1
                        })
                
                # Сохраняем промежуточные результаты после каждой культуры
                if crop_pests:
                    temp_filename = f'pests_{crop_name}_{category}.json'
                    with open(os.path.join(OUTPUT_DIR, temp_filename), 'w', encoding='utf-8') as f:
                        json.dump(crop_pests, f, ensure_ascii=False, indent=4)
                    logger.info(f"Сохранен промежуточный файл {temp_filename}")
            except Exception as e:
                logger.error(f"Ошибка при парсинге вредителей для {crop_name}: {e}")
                # Продолжаем с другими культурами

    # Парсинг сорняков
    logger.info("Парсинг сорняков")
    try:
        crop_weeds = parse_weeds(urls['weeds'])
        logger.debug(f"Извлечено {len(crop_weeds)} сорняков: {[w['name'] for w in crop_weeds]}")
        
        # Сохраняем промежуточные результаты
        if crop_weeds:
            temp_filename = 'weeds_all.json'
            with open(os.path.join(OUTPUT_DIR, temp_filename), 'w', encoding='utf-8') as f:
                json.dump(crop_weeds, f, ensure_ascii=False, indent=4)
            logger.info(f"Сохранен промежуточный файл {temp_filename}")
        
        for weed in crop_weeds:
            weed_id = weed['id']
            desc_id = str(uuid.uuid4())
            img_id = str(uuid.uuid4())

            weeds.append({
                'id': weed_id,
                'name': weed['name'],
                'name_en': weed['name_en'],
                'scientific_name': weed['scientific_name'],
                'is_active': weed['is_active']
            })

            weed_descriptions.append({
                'id': desc_id,
                'weed_id': weed_id,
                'description_ru': weed['description_ru'],
                'description_ua': weed['description_ua'],
                'description_en': weed['description_en'],
                'biological_features_ru': weed['biological_features_ru'],
                'biological_features_ua': weed['biological_features_ua'],
                'biological_features_en': weed['biological_features_en'],
                'harmfulness_ru': weed['harmfulness_ru'],
                'harmfulness_ua': weed['harmfulness_ua'],
                'harmfulness_en': weed['harmfulness_en'],
                'control_measures_ru': weed['control_measures_ru'],
                'control_measures_ua': weed['control_measures_ua'],
                'control_measures_en': weed['control_measures_en'],
                'photo_path': weed['photo_path'],
                'source_urls': weed['source_urls'],
                'version': 1
            })

            # Сорняки могут встречаться на всех культурах
            for crop in ['пшеница', 'ячмень', 'кукуруза', 'овес', 'рапс', 'горчица', 'соя', 'сахарная свекла', 'подсолнечник', 'горох', 'нут', 'лен', 'картофель']:
                weed_crops.append({
                    'weed_id': weed_id,
                    'crops': crop
                })

            if weed['photo_path']:
                weed_images.append({
                    'id': img_id,
                    'weed_id': weed_id,
                    'image_url': weed['source_urls'],
                    'image_path': weed['photo_path'],
                    'version': 1
                })
    except Exception as e:
        logger.error(f"Ошибка при парсинге сорняков: {e}")

    # Логирование количества собранных данных
    logger.debug(f"Общее количество болезней: {len(diseases)}")
    logger.debug(f"Общее количество описаний болезней: {len(disease_descriptions)}")
    logger.debug(f"Общее количество культур для болезней: {len(disease_crops)}")
    logger.debug(f"Общее количество изображений болезней: {len(disease_images)}")
    logger.debug(f"Общее количество вредителей: {len(pests)}")
    logger.debug(f"Общее количество описаний вредителей: {len(pest_descriptions)}")
    logger.debug(f"Общее количество культур для вредителей: {len(pest_crops)}")
    logger.debug(f"Общее количество изображений вредителей: {len(pest_images)}")
    logger.debug(f"Общее количество сорняков: {len(weeds)}")
    logger.debug(f"Общее количество описаний сорняков: {len(weed_descriptions)}")
    logger.debug(f"Общее количество культур для сорняков: {len(weed_crops)}")
    logger.debug(f"Общее количество изображений сорняков: {len(weed_images)}")

    # Сохранение данных в CSV
    # Если данных нет, указываем это в логе, но продолжаем выполнение
    logger.info("Сохранение данных в CSV...")

    # Сохранение информации о болезнях
    if diseases:
        success = save_to_csv(diseases, os.path.join(OUTPUT_DIR, 'diseases.csv'),
                     ['id', 'name', 'name_en', 'scientific_name', 'is_active'])
        if not success:
            # Попробуем сохранить в JSON как запасной вариант
            try:
                with open(os.path.join(OUTPUT_DIR, 'diseases.json'), 'w', encoding='utf-8') as f:
                    json.dump(diseases, f, ensure_ascii=False, indent=4)
                logger.info("Данные о болезнях сохранены в формате JSON")
            except Exception as e:
                logger.error(f"Не удалось сохранить данные о болезнях в JSON: {e}")
    else:
        logger.warning("Нет данных о болезнях для сохранения")

    if disease_descriptions:
        success = save_to_csv(disease_descriptions, os.path.join(OUTPUT_DIR, 'disease_descriptions.csv'),
                     ['id', 'disease_id', 'description_ru', 'description_ua', 'description_en',
                      'symptoms_ru', 'symptoms_ua', 'symptoms_en',
                      'development_conditions_ru', 'development_conditions_ua', 'development_conditions_en',
                      'control_measures_ru', 'control_measures_ua', 'control_measures_en',
                      'photo_path', 'source_urls', 'version'])
        if not success:
            try:
                with open(os.path.join(OUTPUT_DIR, 'disease_descriptions.json'), 'w', encoding='utf-8') as f:
                    json.dump(disease_descriptions, f, ensure_ascii=False, indent=4)
                logger.info("Описания болезней сохранены в формате JSON")
            except Exception as e:
                logger.error(f"Не удалось сохранить описания болезней в JSON: {e}")
    else:
        logger.warning("Нет описаний болезней для сохранения")

    if disease_crops:
        success = save_to_csv(disease_crops, os.path.join(OUTPUT_DIR, 'disease_crops.csv'),
                     ['disease_id', 'crops'])
        if not success:
            try:
                with open(os.path.join(OUTPUT_DIR, 'disease_crops.json'), 'w', encoding='utf-8') as f:
                    json.dump(disease_crops, f, ensure_ascii=False, indent=4)
                logger.info("Связи болезней с культурами сохранены в формате JSON")
            except Exception as e:
                logger.error(f"Не удалось сохранить связи болезней с культурами в JSON: {e}")
    else:
        logger.warning("Нет связей болезней с культурами для сохранения")

    if disease_images:
        success = save_to_csv(disease_images, os.path.join(OUTPUT_DIR, 'disease_images.csv'),
                     ['id', 'disease_id', 'image_url', 'image_path', 'version'])
        if not success:
            try:
                with open(os.path.join(OUTPUT_DIR, 'disease_images.json'), 'w', encoding='utf-8') as f:
                    json.dump(disease_images, f, ensure_ascii=False, indent=4)
                logger.info("Изображения болезней сохранены в формате JSON")
            except Exception as e:
                logger.error(f"Не удалось сохранить изображения болезней в JSON: {e}")
    else:
        logger.warning("Нет изображений болезней для сохранения")

    # Сохранение информации о вредителях
    if pests:
        success = save_to_csv(pests, os.path.join(OUTPUT_DIR, 'vermins.csv'),
                     ['id', 'name', 'name_en', 'scientific_name', 'is_active'])
        if not success:
            try:
                with open(os.path.join(OUTPUT_DIR, 'vermins.json'), 'w', encoding='utf-8') as f:
                    json.dump(pests, f, ensure_ascii=False, indent=4)
                logger.info("Данные о вредителях сохранены в формате JSON")
            except Exception as e:
                logger.error(f"Не удалось сохранить данные о вредителях в JSON: {e}")
    else:
        logger.warning("Нет данных о вредителях для сохранения")

    if pest_descriptions:
        success = save_to_csv(pest_descriptions, os.path.join(OUTPUT_DIR, 'vermin_descriptions.csv'),
                     ['id', 'vermin_id', 'description_ru', 'description_ua', 'description_en',
                      'damage_symptoms_ru', 'damage_symptoms_ua', 'damage_symptoms_en',
                      'biology_ru', 'biology_ua', 'biology_en',
                      'control_measures_ru', 'control_measures_ua', 'control_measures_en',
                      'photo_path', 'source_urls', 'version'])
        if not success:
            try:
                with open(os.path.join(OUTPUT_DIR, 'vermin_descriptions.json'), 'w', encoding='utf-8') as f:
                    json.dump(pest_descriptions, f, ensure_ascii=False, indent=4)
                logger.info("Описания вредителей сохранены в формате JSON")
            except Exception as e:
                logger.error(f"Не удалось сохранить описания вредителей в JSON: {e}")
    else:
        logger.warning("Нет описаний вредителей для сохранения")

    if pest_crops:
        success = save_to_csv(pest_crops, os.path.join(OUTPUT_DIR, 'vermin_crops.csv'),
                     ['vermin_id', 'crops'])
        if not success:
            try:
                with open(os.path.join(OUTPUT_DIR, 'vermin_crops.json'), 'w', encoding='utf-8') as f:
                    json.dump(pest_crops, f, ensure_ascii=False, indent=4)
                logger.info("Связи вредителей с культурами сохранены в формате JSON")
            except Exception as e:
                logger.error(f"Не удалось сохранить связи вредителей с культурами в JSON: {e}")
    else:
        logger.warning("Нет связей вредителей с культурами для сохранения")

    if pest_images:
        success = save_to_csv(pest_images, os.path.join(OUTPUT_DIR, 'vermin_images.csv'),
                     ['id', 'vermin_id', 'image_url', 'image_path', 'version'])
        if not success:
            try:
                with open(os.path.join(OUTPUT_DIR, 'vermin_images.json'), 'w', encoding='utf-8') as f:
                    json.dump(pest_images, f, ensure_ascii=False, indent=4)
                logger.info("Изображения вредителей сохранены в формате JSON")
            except Exception as e:
                logger.error(f"Не удалось сохранить изображения вредителей в JSON: {e}")
    else:
        logger.warning("Нет изображений вредителей для сохранения")

    # Сохранение информации о сорняках
    if weeds:
        success = save_to_csv(weeds, os.path.join(OUTPUT_DIR, 'weeds.csv'),
                     ['id', 'name', 'name_en', 'scientific_name', 'is_active'])
        if not success:
            try:
                with open(os.path.join(OUTPUT_DIR, 'weeds.json'), 'w', encoding='utf-8') as f:
                    json.dump(weeds, f, ensure_ascii=False, indent=4)
                logger.info("Данные о сорняках сохранены в формате JSON")
            except Exception as e:
                logger.error(f"Не удалось сохранить данные о сорняках в JSON: {e}")
    else:
        logger.warning("Нет данных о сорняках для сохранения")

    if weed_descriptions:
        success = save_to_csv(weed_descriptions, os.path.join(OUTPUT_DIR, 'weed_descriptions.csv'),
                     ['id', 'weed_id', 'description_ru', 'description_ua', 'description_en',
                      'biological_features_ru', 'biological_features_ua', 'biological_features_en',
                      'harmfulness_ru', 'harmfulness_ua', 'harmfulness_en',
                      'control_measures_ru', 'control_measures_ua', 'control_measures_en',
                      'photo_path', 'source_urls', 'version'])
        if not success:
            try:
                with open(os.path.join(OUTPUT_DIR, 'weed_descriptions.json'), 'w', encoding='utf-8') as f:
                    json.dump(weed_descriptions, f, ensure_ascii=False, indent=4)
                logger.info("Описания сорняков сохранены в формате JSON")
            except Exception as e:
                logger.error(f"Не удалось сохранить описания сорняков в JSON: {e}")
    else:
        logger.warning("Нет описаний сорняков для сохранения")

    if weed_crops:
        success = save_to_csv(weed_crops, os.path.join(OUTPUT_DIR, 'weed_crops.csv'),
                     ['weed_id', 'crops'])
        if not success:
            try:
                with open(os.path.join(OUTPUT_DIR, 'weed_crops.json'), 'w', encoding='utf-8') as f:
                    json.dump(weed_crops, f, ensure_ascii=False, indent=4)
                logger.info("Связи сорняков с культурами сохранены в формате JSON")
            except Exception as e:
                logger.error(f"Не удалось сохранить связи сорняков с культурами в JSON: {e}")
    else:
        logger.warning("Нет связей сорняков с культурами для сохранения")

    if weed_images:
        success = save_to_csv(weed_images, os.path.join(OUTPUT_DIR, 'weed_images.csv'),
                     ['id', 'weed_id', 'image_url', 'image_path', 'version'])
        if not success:
            try:
                with open(os.path.join(OUTPUT_DIR, 'weed_images.json'), 'w', encoding='utf-8') as f:
                    json.dump(weed_images, f, ensure_ascii=False, indent=4)
                logger.info("Изображения сорняков сохранены в формате JSON")
            except Exception as e:
                logger.error(f"Не удалось сохранить изображения сорняков в JSON: {e}")
    else:
        logger.warning("Нет изображений сорняков для сохранения")

    # Сохранение полного отчета в JSON
    try:
        data = {
            'diseases': {
                'cereals': [d for d in diseases if any(c['crops'] in ['пшеница', 'ячмень', 'овес'] for c in disease_crops if c['disease_id'] == d['id'])],
                'rapeseed': [d for d in diseases if any(c['crops'] == 'рапс' for c in disease_crops if c['disease_id'] == d['id'])],
                'corn': [d for d in diseases if any(c['crops'] == 'кукуруза' for c in disease_crops if c['disease_id'] == d['id'])],
                'soy': [d for d in diseases if any(c['crops'] == 'соя' for c in disease_crops if c['disease_id'] == d['id'])],
                'sugar_beet': [d for d in diseases if any(c['crops'] == 'сахарная свекла' for c in disease_crops if c['disease_id'] == d['id'])],
                'sunflower': [d for d in diseases if any(c['crops'] == 'подсолнечник' for c in disease_crops if c['disease_id'] == d['id'])],
                'pea_nut': [d for d in diseases if any(c['crops'] in ['горох', 'нут'] for c in disease_crops if c['disease_id'] == d['id'])],
                'flax': [d for d in diseases if any(c['crops'] == 'лен' for c in disease_crops if c['disease_id'] == d['id'])],
                'potato': [d for d in diseases if any(c['crops'] == 'картофель' for c in disease_crops if c['disease_id'] == d['id'])],
                'orchard': [d for d in diseases if any(c['crops'] == 'садовые культуры' for c in disease_crops if c['disease_id'] == d['id'])],
                'vineyard': [d for d in diseases if any(c['crops'] == 'виноградники' for c in disease_crops if c['disease_id'] == d['id'])]
            },
            'pests': {
                'cereals': [p for p in pests if any(c['crops'] in ['пшеница', 'ячмень', 'овес'] for c in pest_crops if c['vermin_id'] == p['id'])],
                'rapeseed': [p for p in pests if any(c['crops'] == 'рапс' for c in pest_crops if c['vermin_id'] == p['id'])],
                'corn': [p for p in pests if any(c['crops'] == 'кукуруза' for c in pest_crops if c['vermin_id'] == p['id'])],
                'soy': [p for p in pests if any(c['crops'] == 'соя' for c in pest_crops if c['vermin_id'] == p['id'])],
                'sugar_beet': [p for p in pests if any(c['crops'] == 'сахарная свекла' for c in pest_crops if c['vermin_id'] == p['id'])],
                'sunflower': [p for p in pests if any(c['crops'] == 'подсолнечник' for c in pest_crops if c['vermin_id'] == p['id'])],
                'pea_nut': [p for p in pests if any(c['crops'] in ['горох', 'нут'] for c in pest_crops if c['vermin_id'] == p['id'])],
                'flax': [p for p in pests if any(c['crops'] == 'лен' for c in pest_crops if c['vermin_id'] == p['id'])],
                'potato': [p for p in pests if any(c['crops'] == 'картофель' for c in pest_crops if c['vermin_id'] == p['id'])],
                'orchard': [p for p in pests if any(c['crops'] == 'садовые культуры' for c in pest_crops if c['vermin_id'] == p['id'])],
                'vineyard': [p for p in pests if any(c['crops'] == 'виноградники' for c in pest_crops if c['vermin_id'] == p['id'])]
            },
            'weeds': weeds
        }

        with open(os.path.join(OUTPUT_DIR, 'betaren_data.json'), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f"JSON-отчет сохранен в {os.path.join(OUTPUT_DIR, 'betaren_data.json')}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении JSON-отчета: {e}")

    logger.info("Парсинг завершен!")

if __name__ == "__main__":
    main()