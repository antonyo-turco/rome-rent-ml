"""
Data Cleanup Script for Rome Rent ML Project
This script processes the raw Rome rent data and outputs a cleaned CSV file.
"""

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer


def clean_price(row, warning=False):
    """Clean the 'prezzo' column by removing /mese and converting to float."""
    price = row['prezzo']
    if isinstance(price, str):
        price = price.replace('/mese', '')
        price = price.replace('\x80', '').replace(".", "").strip()
        try:
            return float(price)
        except ValueError:
            if warning:
                print(f"Warning: Could not convert price '{row['prezzo']}' to float.")
            return None
    return price


def clean_spese_condominio(row, warning=False):
    """Clean and convert spese condominio to float."""
    spese = row['spese condominio']
    if isinstance(spese, str):
        spese = spese.replace('\x80', '').replace(".", "").strip()
        if spese.lower() in ['n.d.', 'nessuna', 'nessun', 'nessuno', 'non indicato', 
                             'Nessuna spesa condominiale', 'nessuna spesa condominiale', 
                             'nessun dato', '0', 'zero', 'null', 'nulla', 'nil', '-']:
            return 0.0
        try:
            return float(spese)
        except ValueError:
            if warning:
                print(f"Warning: Could not convert spese condominio '{row['spese condominio']}' to float.")
            return None
    return spese


def clean_mq(row, warning=False):
    """Clean and convert m2 to float."""
    mq = row['m2']
    if isinstance(mq, str):
        mq = mq.replace('m²', '').strip()
        try:
            return float(mq)
        except ValueError:
            if warning:
                print(f"Warning: Could not convert m2 '{row['m2']}' to float.")
            return None
    return mq


def clean_rooms(row, warning=False):
    """Parse rooms column handling 5+ notation."""
    stanze = row['stanze']
    if isinstance(stanze, str):
        stanze = stanze.strip()
        if stanze == '5+':
            return pd.Series({'rooms': 5, 'more_than_5_rooms': True})
        else:
            try:
                rooms = int(stanze)
                return pd.Series({'rooms': rooms, 'more_than_5_rooms': False})
            except ValueError:
                if warning:
                    print(f"Warning: Could not convert rooms '{row['stanze']}' to int.")
                return pd.Series({'rooms': None, 'more_than_5_rooms': None})
    return pd.Series({'rooms': stanze, 'more_than_5_rooms': False})


def clean_floor(row, warning=False):
    """Parse floor information including elevator and accessibility."""
    piano = row['piano']
    ascensore = False
    accesso_disabili = False
    piano_rialzato = False
    multi_floor = False
    floor_value = None
    totale_piani_edificio = row.get('totale piani edificio', None)
    ultimo_piano = False
    
    if isinstance(totale_piani_edificio, str):
        try:
            totale_piani_edificio = int(totale_piani_edificio.split()[0])
        except ValueError:
            if warning:
                print(f"Warning: Could not convert total floors '{row['totale piani edificio']}' to int.")
            totale_piani_edificio = None
    
    if isinstance(piano, str):
        piano = piano.lower().strip()
        if 'ascensore' in piano:
            ascensore = True
            piano = piano.replace('ascensore', '').strip()
        if 'accesso disabili' in piano:
            accesso_disabili = True
            piano = piano.replace('accesso disabili', '').strip()
        if 'seminterrato' in piano:
            floor_value = -1
        elif 'piano terra' in piano or 'terra' in piano:
            floor_value = 0
        elif 'piano rialzato' in piano:
            floor_value = 0
            piano_rialzato = True
        elif '°' in piano:
            try:
                parts = piano.replace(',', ' ').split()
                for part in parts:
                    if '°' in part:
                        floor_value = int(part.replace('°', ''))
                        break
            except ValueError:
                if warning:
                    print(f"Warning: Could not convert floor value '{row['piano']}' to int.")
                floor_value = None
            if totale_piani_edificio is not None and floor_value is not None:
                if floor_value >= totale_piani_edificio:
                    ultimo_piano = True
        else:
            if len(piano) > 2 and piano[1] != '°':
                multi_floor = True
    
    return pd.Series({
        'floor': floor_value,
        'ascensore': ascensore,
        'accesso_disabili': accesso_disabili,
        'piano_rialzato': piano_rialzato,
        'multi_floor': multi_floor,
        'totale_piani_edificio': totale_piani_edificio,
        'ultimo_piano': ultimo_piano
    })


def clean_contratto(row, warning=False):
    """Parse contract type information."""
    contratto = row['contratto']
    affitto = False
    affitto_libero = False
    affitto_durata_minima = None
    affitto_durata_rinnovo = None
    affitto_concordato = False
    affitto_transitorio = False
    affitto_studenti = False
    affitto_riscatto = False
    immobile_a_reddito = False
    
    if isinstance(contratto, str):
        contratto = contratto.lower()
        if 'affitto' in contratto:
            affitto = True
        if 'libero' in contratto:
            affitto_libero = True
        if 'concordato' in contratto:
            affitto_concordato = True
        if 'transitorio' in contratto:
            affitto_transitorio = True
        if 'studenti' in contratto:
            affitto_studenti = True
        if 'riscatto' in contratto:
            affitto_riscatto = True
        if 'immobile a reddito' in contratto:
            immobile_a_reddito = True
        
        match = re.search(r'(\d+)\+(\d+)', contratto)
        if match:
            affitto_durata_minima = int(match.group(1))
            affitto_durata_rinnovo = int(match.group(2))
        else:
            affitto_durata_minima = 0
            affitto_durata_rinnovo = 0
    
    return pd.Series({
        'affitto': affitto,
        'affitto_libero': affitto_libero,
        'affitto_durata_minima': affitto_durata_minima,
        'affitto_durata_rinnovo': affitto_durata_rinnovo,
        'affitto_concordato': affitto_concordato,
        'affitto_transitorio': affitto_transitorio,
        'affitto_studenti': affitto_studenti,
        'affitto_riscatto': affitto_riscatto,
        'immobile_a_reddito': immobile_a_reddito
    })


def clean_locali(row, warning=False):
    """Parse locali column for rooms and kitchen type."""
    locali = row['locali']
    totale_locali = 0
    camere_da_letto = 0
    altri_locali = 0
    tipo_cucina = ""
    campo_da_tennis = False
    
    if isinstance(locali, str):
        if "campo da tennis" in locali:
            campo_da_tennis = True
        
        match_totale = re.search(r'(\d+|\d+\+)', locali)
        if match_totale:
            totale_locali = int(match_totale.group(0).replace('+', ''))
        
        match_locali = re.search(r'(\d+) camere da letto, (\d+) altri', locali)
        if match_locali:
            camere_da_letto = int(match_locali.group(1))
            altri_locali = int(match_locali.group(2))
        
        match_cucina = re.search(r'cucina ([a-z\s]+)', locali)
        if match_cucina:
            tipo_cucina = match_cucina.group(1).strip()
    
    return pd.Series({
        'totale_locali': totale_locali,
        'camere_da_letto': camere_da_letto,
        'altri_locali': altri_locali,
        'tipo_cucina': tipo_cucina,
        'campo_da_tennis': campo_da_tennis
    })


def clean_bathrooms(row, warning=False):
    """Parse bathrooms handling 3+ notation."""
    bagni = row['bagni']
    more_than_3_bathrooms = False
    
    if isinstance(bagni, str):
        bagni = bagni.strip()
        if bagni == '3+':
            more_than_3_bathrooms = True
            return pd.Series({'bathrooms': 3, 'more_than_3_bathrooms': more_than_3_bathrooms})
        else:
            try:
                bathrooms = int(bagni)
                return pd.Series({'bathrooms': bathrooms, 'more_than_3_bathrooms': more_than_3_bathrooms})
            except ValueError:
                if warning:
                    print(f"Warning: Could not convert bathrooms '{row['bagni']}' to int.")
                return pd.Series({'bathrooms': None, 'more_than_3_bathrooms': more_than_3_bathrooms})
    return pd.Series({'bathrooms': bagni, 'more_than_3_bathrooms': more_than_3_bathrooms})


def parse_parking(row):
    """Parse parking information."""
    posti_auto = row['Posti Auto']
    garage_box = 0
    esterno = 0
    parcheggio_comune = 0
    box_privato = 0

    if isinstance(posti_auto, str):
        garage_box_matches = re.findall(r'(\d+) in garage/box', posti_auto)
        esterno_matches = re.findall(r'(\d+) all\'esterno', posti_auto)
        parcheggio_comune_matches = re.findall(r'(\d+) in parcheggio/garage comune', posti_auto)
        box_privato_matches = re.findall(r'(\d+) in box privato/box in garage', posti_auto)

        garage_box = sum(map(int, garage_box_matches))
        esterno = sum(map(int, esterno_matches))
        parcheggio_comune = sum(map(int, parcheggio_comune_matches))
        box_privato = sum(map(int, box_privato_matches))

    return pd.Series({
        'garage_box': garage_box,
        'esterno': esterno,
        'parcheggio_comune': parcheggio_comune,
        'box_privato': box_privato
    })


def parse_stato(row):
    """Parse stato column into condition and renovation status."""
    stato = row['stato']
    stato_condition = 'Unknown'
    stato_renovation = None
    
    if isinstance(stato, str):
        parts = [part.strip() for part in stato.split('/')]
        if len(parts) == 2:
            stato_condition = parts[0]
            stato_renovation = parts[1]
        elif len(parts) == 1:
            stato_condition = parts[0]
    
    return pd.Series({
        'stato_condition': stato_condition,
        'stato_renovation': stato_renovation
    })


def parse_tipologia(row):
    """Parse property type information."""
    tipologia = row['tipologia']
    property_types = {
        'appartamento': False,
        'attico': False,
        'villa_unifamiliare': False,
        'villa_bifamiliare': False,
        'villa_plurifamiliare': False,
        'open_space': False,
        'mansarda': False,
        'loft': False,
        'palazzo_edificio': False,
        'casale': False,
        'terratetto_unifamiliare': False,
        'terratetto_plurifamiliare': False,
        'classe_signorile': False,
        'classe_media': False,
        'classe_economica': False,
        'immobile_di_lusso': False,
        'intera_proprieta': False,
        'parziale_proprieta': False,
        'diritto_di_superficie': False,
        'nuda_proprieta': False,
        'usufrutto': False,
        'multiproprieta': False
    }

    if isinstance(tipologia, str):
        tipologia = tipologia.lower()
        for key in property_types.keys():
            if key.replace('_', ' ') in tipologia:
                property_types[key] = True

    return pd.Series(property_types)


def clean_disponibilita(row, warning=False):
    """Parse disponibilita column."""
    disponibilita = row['disponibilità']
    if isinstance(disponibilita, str):
        disponibilita = disponibilita.lower().strip()
        if 'libero' in disponibilita:
            return True
        else:
            return False
    return disponibilita


def clean_anno_di_costruzione(row, warning=False):
    """Parse construction year."""
    anno_di_costruzione = row['anno di costruzione']
    if isinstance(anno_di_costruzione, str):
        anno_di_costruzione = anno_di_costruzione.strip()
        try:
            return int(anno_di_costruzione)
        except ValueError:
            if warning:
                print(f"Warning: Could not convert anno_di_costruzione '{row['anno_di_costruzione']}' to int.")
            return None
    return anno_di_costruzione


def clean_riscaldamento(row, warning=False):
    """Parse heating system information."""
    riscaldamento = row['riscaldamento']
    autonomo = False
    centralizzato = False
    radiatori = False
    pavimento = False
    aria = False
    stufa = False
    gas = False
    metano = False
    gpl = False
    gasolio = False
    pompa_di_calore = False
    teleriscaldamento = False
    fotovoltaico = False
    solare = False
    elettrico = False
    pellet = False
    
    if isinstance(riscaldamento, str):
        riscaldamento = riscaldamento.lower()
        if 'autonomo' in riscaldamento:
            autonomo = True
        if 'centralizzato' in riscaldamento:
            centralizzato = True
        if 'radiatori' in riscaldamento:
            radiatori = True
        if 'pavimento' in riscaldamento:
            pavimento = True
        if 'aria' in riscaldamento:
            aria = True
        if 'stufa' in riscaldamento:
            stufa = True
        if 'metano' in riscaldamento:
            metano = True
        if 'gas' in riscaldamento and 'gasolio' not in riscaldamento:
            gas = True
        if 'gpl' in riscaldamento:
            gpl = True
        if 'gasolio' in riscaldamento:
            gasolio = True
        if 'pompa di calore' in riscaldamento:
            pompa_di_calore = True
        if 'teleriscaldamento' in riscaldamento:
            teleriscaldamento = True
        if 'fotovoltaico' in riscaldamento:
            fotovoltaico = True
        if 'solare' in riscaldamento:
            solare = True
        if 'elettrica' in riscaldamento or 'elettrico' in riscaldamento:
            elettrico = True
        if 'pellet' in riscaldamento:
            pellet = True
    
    return pd.Series({
        'riscaldamento_autonomo': autonomo,
        'riscaldamento_centralizzato': centralizzato,
        'riscaldamento_radiatori': radiatori,
        'riscaldamento_pavimento': pavimento,
        'riscaldamento_aria': aria,
        'riscaldamento_stufa': stufa,
        'riscaldamento_gas': gas,
        'riscaldamento_metano': metano,
        'riscaldamento_gpl': gpl,
        'riscaldamento_gasolio': gasolio,
        'riscaldamento_pompa_calore': pompa_di_calore,
        'riscaldamento_teleriscaldamento': teleriscaldamento,
        'riscaldamento_fotovoltaico': fotovoltaico,
        'riscaldamento_solare': solare,
        'riscaldamento_elettrico': elettrico,
        'riscaldamento_pellet': pellet
    })


def clean_climatizzatore(row, warning=False):
    """Parse air conditioning information."""
    climatizzatore = row['Climatizzatore']
    autonomo = False
    centralizzato = False
    predisposizione = False
    freddo = False
    caldo = False
    assente = False
    
    if isinstance(climatizzatore, str):
        climatizzatore = climatizzatore.lower()
        if 'autonomo' in climatizzatore:
            autonomo = True
        if 'centralizzato' in climatizzatore:
            centralizzato = True
        if 'predisposizione' in climatizzatore:
            predisposizione = True
        if 'assente' in climatizzatore:
            assente = True
        if 'freddo' in climatizzatore:
            freddo = True
        if 'caldo' in climatizzatore:
            caldo = True
    
    return pd.Series({
        'climatizzatore_autonomo': autonomo,
        'climatizzatore_centralizzato': centralizzato,
        'climatizzatore_predisposizione': predisposizione,
        'climatizzatore_freddo': freddo,
        'climatizzatore_caldo': caldo,
        'climatizzatore_assente': assente
    })


def clean_efficienza_energetica(row, warning=False):
    """Parse energy efficiency information."""
    efficienza = row['Efficienza energetica']
    classe = None
    classe_numerica = None
    consumo_kwh = None
    
    classe_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    
    if isinstance(efficienza, str):
        efficienza = efficienza.strip()
        if len(efficienza) > 0 and efficienza[0].isalpha():
            classe = efficienza[0].upper()
            classe_numerica = classe_mapping.get(classe, None)
        
        match = re.search(r'(\d+(?:[.,]\d+)?)', efficienza)
        if match:
            try:
                consumo_str = match.group(1).replace(',', '.')
                consumo_kwh = float(consumo_str)
            except ValueError:
                if warning:
                    print(f"Warning: Could not convert efficienza energetica '{efficienza}' to float.")
                consumo_kwh = None
    
    return pd.Series({
        'efficienza_classe': classe,
        'efficienza_classe_numerica': classe_numerica,
        'efficienza_consumo_kwh': consumo_kwh
    })


def parse_altre_caratteristiche(row):
    """Parse additional property features."""
    caratteristiche_str = row['altre caratteristiche']
    features = {
        'terrazza': False,
        'reception': False,
        'idromassaggio': False,
        'ascensore': False,
        'esposizione_sud': False,
        'impianto_tv': False,
        'portiere': False,
        'balconi': False,
        'taverna': False,
        'cucina_arredata': False,
        'videocitofono': False,
        'box_garage': False,
        'piscina': False,
        'cantina': False,
        'caminetto': False,
        'esposizione_doppia': False,
        'arredato': False,
        'giardino': False,
        'armadio_a_muro': False,
        'piano_terra': False,
        'mansarda': False,
        'esposizione_interna': False,
        'cancello_elettrico': False,
        'esposizione_esterna': False,
        'infissi_qualità': False,
        'accesso_disabili': False,
        'esposizione_est': False,
        'campo_da_tennis': False,
        'impianto_di_allarme': False,
        'fibra_ottica': False,
        'porta_blindata': False
    }

    if isinstance(caratteristiche_str, str):
        caratteristiche_lower = [feature.strip().lower() for feature in caratteristiche_str.split(',')]
        caratteristiche_str_lower = ' '.join(caratteristiche_lower)
        
        if any(x in caratteristiche_str_lower for x in ['balconi', '1 balcone', '2 balconi', '3 balconi', '4 balconi', '5 balconi', '6 balconi', '8 balconi', '18 balconi']):
            features['balconi'] = True
        if any(x in caratteristiche_str_lower for x in ['impianto tv singolo', 'impianto tv centralizzato', 'impianto tv con parabola']):
            features['impianto_tv'] = True
        if any(x in caratteristiche_str_lower for x in ['portiere intera giornata', 'portiere mezza giornata']):
            features['portiere'] = True
        if any(x in caratteristiche_str_lower for x in ['giardino privato', 'giardino comune', 'giardino privato e comune']):
            features['giardino'] = True
        if any(x in caratteristiche_str_lower for x in ['solo cucina arredata', 'parzialmente arredato']):
            features['cucina_arredata'] = True
        if any(x in caratteristiche_str_lower for x in ['box', 'garage', 'box/garage privato', '1 in box/garage privato']):
            features['box_garage'] = True
        if any(x in caratteristiche_str_lower for x in ['infissi', 'doppio vetro', 'triplo vetro']):
            features['infissi_qualità'] = True
        if 'porta blindata' in caratteristiche_str_lower:
            features['porta_blindata'] = True
        if 'terrazza' in caratteristiche_str_lower:
            features['terrazza'] = True
        if 'reception' in caratteristiche_str_lower:
            features['reception'] = True
        if 'idromassaggio' in caratteristiche_str_lower:
            features['idromassaggio'] = True
        if 'ascensore' in caratteristiche_str_lower or 'con ascensore' in caratteristiche_str_lower:
            features['ascensore'] = True
        if 'esposizione sud' in caratteristiche_str_lower:
            features['esposizione_sud'] = True
        if 'taverna' in caratteristiche_str_lower:
            features['taverna'] = True
        if 'videocitofono' in caratteristiche_str_lower:
            features['videocitofono'] = True
        if 'piscina' in caratteristiche_str_lower:
            features['piscina'] = True
        if 'cantina' in caratteristiche_str_lower:
            features['cantina'] = True
        if 'caminetto' in caratteristiche_str_lower:
            features['caminetto'] = True
        if 'esposizione doppia' in caratteristiche_str_lower:
            features['esposizione_doppia'] = True
        if 'arredato' in caratteristiche_str_lower:
            features['arredato'] = True
        if 'armadio a muro' in caratteristiche_str_lower:
            features['armadio_a_muro'] = True
        if 'piano terra' in caratteristiche_str_lower:
            features['piano_terra'] = True
        if 'mansarda' in caratteristiche_str_lower:
            features['mansarda'] = True
        if 'esposizione interna' in caratteristiche_str_lower:
            features['esposizione_interna'] = True
        if 'cancello elettrico' in caratteristiche_str_lower:
            features['cancello_elettrico'] = True
        if 'esposizione esterna' in caratteristiche_str_lower:
            features['esposizione_esterna'] = True
        if 'accesso disabili' in caratteristiche_str_lower or 'con accesso disabili' in caratteristiche_str_lower:
            features['accesso_disabili'] = True
        if 'esposizione est' in caratteristiche_str_lower:
            features['esposizione_est'] = True
        if 'campo da tennis' in caratteristiche_str_lower:
            features['campo_da_tennis'] = True
        if 'impianto di allarme' in caratteristiche_str_lower:
            features['impianto_di_allarme'] = True
        if 'fibra ottica' in caratteristiche_str_lower:
            features['fibra_ottica'] = True

    return pd.Series(features)


def parse_description(row):
    """Parse description column for specific keywords."""
    description = row['description']
    features = {
        "metro": 0,
        "stazione": 0,
        "universita": 0,
        "ospedale": 0,
        "parco": 0,
        "doccia": 0,
        "vasca": 0,
        "luminoso": 0,
        "silenzioso": 0,
        "guardaroba": 0,
        "mercato": 0,
    }

    if isinstance(description, str):
        description = description.lower()
        if "metro" or "metropolitana" in description:
            features["metro"] = 1
        if "stazione" or "treno" in description:
            features["stazione"] = 1
        if "universita" or "università" in description:
            features["universita"] = 1
        if "ospedale" in description:
            features["ospedale"] = 1
        if "parco" in description:
            features["parco"] = 1
        if "doccia" in description:
            features["doccia"] = 1
        if "vasca" in description:
            features["vasca"] = 1
        if "luminoso" or "luminosa" in description:
            features["luminoso"] = 1
        if "silenzioso" or "silenziosa" in description:
            features["silenzioso"] = 1
        if "guardaroba" in description:
            features["guardaroba"] = 1
        if "mercato" or "supermercato" or "super market" or "supermarket" in description:
            features["mercato"] = 1

    return pd.Series(features)


def clean_cauzione(row, warning=False):
    """Parse deposit (cauzione) as float."""
    cauzione = row['cauzione']
    if isinstance(cauzione, str):
        cauzione = cauzione.replace('\x80', '').replace(".", "").strip()
        try:
            return float(cauzione)
        except ValueError:
            if warning:
                print(f"Warning: Could not convert cauzione '{row['cauzione']}' to float.")
            return None
    return cauzione


def main():
    """Main function to clean Rome rent data and export to CSV."""
    print("Loading raw data...")
    data = pd.read_csv("italy-house-prices/rome_rents_raw.csv", encoding="latin1")
    print(f"Loaded {len(data)} rows")
    
    # Rename columns to English
    column_rename_map = {
        'prezzo': 'price',
        'spese condominio': 'condo_fees',
        'm2': 'square_meters',
        'stanze': 'rooms',
        'piano': 'floor',
        'totale piani edificio': 'total_floors',
        'contratto': 'contract',
        'locali': 'rooms_details',
        'bagni': 'bathrooms',
        'Posti Auto': 'parking_spaces',
        'stato': 'condition',
        'tipologia': 'property_type',
        'disponibilità': 'availability',
        'anno di costruzione': 'year_built',
        'riscaldamento': 'heating',
        'Climatizzatore': 'air_conditioning',
        'Efficienza energetica': 'energy_efficiency',
        'altre caratteristiche': 'other_features',
        'cauzione': 'deposit',
        'quartiere': 'neighborhood',
        'description': 'description'
    }
    data.rename(columns=column_rename_map, inplace=True)
    
    # Create empty dataframe for cleaned data
    cleaned_data = pd.DataFrame()
    
    # Apply all cleaning functions
    print("Cleaning price...")
    cleaned_data['prezzo'] = data.apply(clean_price, axis=1)
    
    print("Cleaning spese condominio...")
    cleaned_data['spese_condominio'] = data.apply(clean_spese_condominio, axis=1)
    
    print("Cleaning m2...")
    cleaned_data['m2'] = data.apply(clean_mq, axis=1)
    
    print("Cleaning rooms...")
    rooms_data = data.apply(clean_rooms, axis=1)
    cleaned_data = pd.concat([cleaned_data, rooms_data], axis=1)
    
    print("Cleaning floor...")
    floor_data = data.apply(clean_floor, axis=1)
    cleaned_data = pd.concat([cleaned_data, floor_data], axis=1)
    
    print("Cleaning contract...")
    contract_data = data.apply(clean_contratto, axis=1)
    cleaned_data = pd.concat([cleaned_data, contract_data], axis=1)
    
    print("Cleaning locali...")
    locali_data = data.apply(clean_locali, axis=1)
    cleaned_data = pd.concat([cleaned_data, locali_data], axis=1)
    tipo_cucina_dummies = pd.get_dummies(cleaned_data['tipo_cucina'], prefix='cucina')
    cleaned_data = pd.concat([cleaned_data, tipo_cucina_dummies], axis=1)
    
    print("Cleaning bathrooms...")
    bathrooms_data = data.apply(clean_bathrooms, axis=1)
    bathrooms_per_locali = bathrooms_data['bathrooms'] / cleaned_data['totale_locali']
    bathrooms_data['bathrooms_per_locali'] = bathrooms_per_locali
    cleaned_data = pd.concat([cleaned_data, bathrooms_data], axis=1)
    
    print("Parsing parking...")
    temp_parking_data = data.apply(parse_parking, axis=1)
    cleaned_data = pd.concat([cleaned_data, temp_parking_data], axis=1)
    cleaned_data['has_garage_box'] = cleaned_data['garage_box'] > 0
    cleaned_data['has_esterno'] = cleaned_data['esterno'] > 0
    cleaned_data['has_parcheggio_comune'] = cleaned_data['parcheggio_comune'] > 0
    cleaned_data['has_box_privato'] = cleaned_data['box_privato'] > 0
    
    print("Parsing stato...")
    stato_data = data.apply(parse_stato, axis=1)
    cleaned_data = pd.concat([cleaned_data, stato_data], axis=1)
    stato_condition_dummies = pd.get_dummies(cleaned_data['stato_condition'], prefix='stato_condition')
    stato_renovation_dummies = pd.get_dummies(cleaned_data['stato_renovation'], prefix='stato_renovation')
    cleaned_data = pd.concat([cleaned_data, stato_condition_dummies, stato_renovation_dummies], axis=1)
    
    print("Parsing tipologia...")
    tipologia_data = data.apply(parse_tipologia, axis=1)
    cleaned_data = pd.concat([cleaned_data, tipologia_data], axis=1)
    
    print("Cleaning disponibilita...")
    cleaned_data['disponibilita'] = data.apply(clean_disponibilita, axis=1)
    
    print("Cleaning anno di costruzione...")
    anno_data = data.apply(clean_anno_di_costruzione, axis=1)
    cleaned_data['anno_di_costruzione'] = anno_data
    
    print("Cleaning riscaldamento...")
    riscaldamento_data = data.apply(clean_riscaldamento, axis=1)
    cleaned_data = pd.concat([cleaned_data, riscaldamento_data], axis=1)
    
    print("Cleaning climatizzatore...")
    climatizzatore_data = data.apply(clean_climatizzatore, axis=1)
    cleaned_data = pd.concat([cleaned_data, climatizzatore_data], axis=1)
    
    print("Cleaning efficienza energetica...")
    efficienza_data = data.apply(clean_efficienza_energetica, axis=1)
    cleaned_data = pd.concat([cleaned_data, efficienza_data], axis=1)
    
    print("Parsing altre caratteristiche...")
    altre_caratteristiche_data = data.apply(parse_altre_caratteristiche, axis=1)
    cleaned_data = pd.concat([cleaned_data, altre_caratteristiche_data], axis=1)
    
    print("Adding quartiere...")
    cleaned_data['quartiere'] = data['quartiere']
    quartiere_dummies = pd.get_dummies(cleaned_data['quartiere'], prefix='quartiere')
    cleaned_data = pd.concat([cleaned_data, quartiere_dummies], axis=1)
    
    print("Parsing description...")
    description_data = data.apply(parse_description, axis=1)
    cleaned_data = pd.concat([cleaned_data, description_data], axis=1)
    
    print("Cleaning cauzione...")
    cleaned_data['cauzione'] = data.apply(clean_cauzione, axis=1)
    
    # drop all with price or m2 as NaN
    cleaned_data = cleaned_data.dropna(subset=['prezzo', 'm2'])
    
    # Reset index
    cleaned_data = cleaned_data.reset_index(drop=True)
    
    # Remove non-encoded string columns
    to_remove = ['tipo_cucina', 'stato_condition', 'stato_renovation', 'efficienza_classe', 'quartiere']
    for col in to_remove:
        if col in cleaned_data.columns:
            cleaned_data = cleaned_data.drop(col, axis=1)
    
    print(f"\nFinal cleaned dataset shape: {cleaned_data.shape}")
    
    # Save to CSV
    output_file = "italy-house-prices/rome_rents_clean_our_version.csv"
    cleaned_data.to_csv(output_file, index=False)
    print(f"\nCleaned data saved to: {output_file}")


if __name__ == "__main__":
    main()
