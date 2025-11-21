"""
Data Cleanup Script for Rome Rent ML Project
This script processes the raw Rome rent data and outputs a cleaned CSV file.
"""

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer


def clean_price(row, warning=False):
    """Clean the 'price' column by removing /mese and converting to float."""
    price = row['price']
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
    spese = row['condo_fees']
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
    mq = row['square_meters']
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
    stanze = row['rooms']
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
    piano = row['floor']
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
        'lift': ascensore,
        'disabled_access': accesso_disabili,
        'raised_floor': piano_rialzato,
        'multi_floor': multi_floor,
        'total_floors_building': totale_piani_edificio,
        'last_floor': ultimo_piano
    })


def clean_contratto(row, warning=False):
    """Parse contract type information."""
    contratto = row['contract']
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
        'rent': affitto,
        'free_rent': affitto_libero,
        'rent_min_duration': affitto_durata_minima,
        'rent_renewal_duration': affitto_durata_rinnovo,
        'controlled_rent': affitto_concordato,
        'short_term_rent': affitto_transitorio,
        'student_rent': affitto_studenti,
        'buyout_rent': affitto_riscatto,
        'income_property': immobile_a_reddito
    })


def clean_locali(row, warning=False):
    """Parse locali column for rooms and kitchen type."""
    locali = row['rooms_details']
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
        'room_total': totale_locali,
        'bedrooms': camere_da_letto,
        'other_rooms': altri_locali,
        'kitchen_type': tipo_cucina,
        'tennis_court': campo_da_tennis
    })


def clean_bathrooms(row, warning=False):
    """Parse bathrooms handling 3+ notation."""
    bagni = row['bathrooms']
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
    posti_auto = row['parking_spaces']
    garage_box = 0
    outdoor_parking = 0
    common_parking = 0
    private_box = 0

    if isinstance(posti_auto, str):
        garage_box_matches = re.findall(r'(\d+) in garage/box', posti_auto)
        outdoor_parking_matches = re.findall(r'(\d+) all\'esterno', posti_auto)
        common_parking_matches = re.findall(r'(\d+) in parcheggio/garage comune', posti_auto)
        private_box_matches = re.findall(r'(\d+) in box privato/box in garage', posti_auto)

        garage_box = sum(map(int, garage_box_matches))
        outdoor_parking = sum(map(int, outdoor_parking_matches))
        common_parking = sum(map(int, common_parking_matches))
        private_box = sum(map(int, private_box_matches))

    return pd.Series({
        'garage_box': garage_box,
        'outdoor_parking': outdoor_parking,
        'common_parking': common_parking,
        'private_box': private_box
    })


def parse_stato(row):
    """Parse stato column into condition and renovation status."""
    stato = row['condition']
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


def parse_housetype(row):
    """Parse property type information (English keys)."""
    # support both renamed column 'property_type' and original 'tipologia'
    housetype = row.get('property_type', row.get('tipologia', None))

    property_types = {
        'condominium': False,
        'penthouse': False,
        'single_family_villa': False,
        'semi_detached_villa': False,
        'multi_family_villa': False,
        'open_space': False,
        'attic': False,
        'loft': False,
        'building': False,
        'farmhouse': False,
        'terraced_house_single': False,
        'terraced_house_multi': False,
        'prestigious_class': False,
        'middle_class': False,
        'economical_class': False,
        'luxury_property': False,
        'full_ownership': False,
        'partial_ownership': False,
        'right_of_surface': False,
        'bare_ownership': False,
        'usufruct': False,
        'timeshare': False
    }

    # map common Italian phrases to the English keys above
    italian_to_english = {
        'condominio': 'condominium',
        'attico': 'penthouse',
        'villa unifamiliare': 'single_family_villa',
        'villa bifamiliare': 'semi_detached_villa',
        'villa plurifamiliare': 'multi_family_villa',
        'open space': 'open_space',
        'mansarda': 'attic',
        'loft': 'loft',
        'palazzo': 'building',
        'edificio': 'building',
        'casale': 'farmhouse',
        'terratetto unifamiliare': 'terraced_house_single',
        'terratetto plurifamiliare': 'terraced_house_multi',
        'classe signorile': 'prestigious_class',
        'classe media': 'middle_class',
        'classe economica': 'economical_class',
        'immobile di lusso': 'luxury_property',
        'intera proprietà': 'full_ownership',
        'parziale proprietà': 'partial_ownership',
        'diritto di superficie': 'right_of_surface',
        'nuda proprietà': 'bare_ownership',
        'usufrutto': 'usufruct',
        'multiproprietà': 'timeshare',
        'multiproprieta': 'timeshare'
    }

    if isinstance(housetype, str):
        lower_ht = housetype.lower()
        for it_term, en_key in italian_to_english.items():
            if it_term in lower_ht:
                property_types[en_key] = True

    return pd.Series(property_types)


def clean_disponibilita(row, warning=False):
    """Parse disponibilita column."""
    disponibilita = row['availability']
    if isinstance(disponibilita, str):
        disponibilita = disponibilita.lower().strip()
        if 'libero' in disponibilita:
            return True
        else:
            return False
    return disponibilita


def clean_anno_di_costruzione(row, warning=False):
    """Parse construction year."""
    anno_di_costruzione = row['year_built']
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
    riscaldamento = row['heating']
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
        'heating_autonomous': autonomo,
        'heating_centralized': centralizzato,
        'heating_radiators': radiatori,
        'heating_floor': pavimento,
        'heating_air': aria,
        'heating_stove': stufa,
        'heating_gas': gas,
        'heating_methane': metano,
        'heating_gpl': gpl,
        'heating_diesel': gasolio,
        'heating_heat_pump': pompa_di_calore,
        'heating_district_heating': teleriscaldamento,
        'heating_photovoltaic': fotovoltaico,
        'heating_solar': solare,
        'heating_electric': elettrico,
        'heating_pellet': pellet
    })


def clean_air_conditioning(row, warning=False):
    """Parse air conditioning information."""
    climatizzatore = row['air_conditioning']
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
        'air_conditioning_autonomous': autonomo,
        'air_conditioning_centralized': centralizzato,
        'air_conditioning_prearranged': predisposizione,
        'air_conditioning_cold': freddo,
        'air_conditioning_hot': caldo,
        'air_conditioning_absent': assente
    })


def clean_energy_efficiency(row, warning=False):
    """Parse energy efficiency information."""
    efficienza = row['energy_efficiency']
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
        'energy_efficiency_class': classe,
        'energy_efficiency_consumption_kwh': consumo_kwh
    })


def parse_additional_features(row):
    """Parse additional property features."""
    caratteristiche_str = row['other_features']
    features = {
        'terrace': False,
        'reception': False,
        'jacuzzi': False,
        'elevator': False,
        'south_facing': False,
        'tv_system': False,
        'concierge': False,
        'balconies': False,
        'tavern': False,
        'furnished_kitchen': False,
        'video_intercom': False,
        'garage_box': False,
        'swimming_pool': False,
        'cellar': False,
        'fireplace': False,
        'double_exposure': False,
        'furnished': False,
        'garden': False,
        'built_in_wardrobe': False,
        'ground_floor': False,
        'attic': False,
        'internal_exposure': False,
        'electric_gate': False,
        'external_exposure': False,
        'high_quality_windows': False,
        'disabled_access': False,
        'east_facing': False,
        'tennis_court': False,
        'alarm_system': False,
        'fiber_optic': False,
        'security_door': False
    }

    if isinstance(caratteristiche_str, str):
        caratteristiche_lower = [feature.strip().lower() for feature in caratteristiche_str.split(',')]
        caratteristiche_str_lower = ' '.join(caratteristiche_lower)
        
        if any(x in caratteristiche_str_lower for x in ['balconi', '1 balcone', '2 balconi', '3 balconi', '4 balconi', '5 balconi', '6 balconi', '8 balconi', '18 balconi']):
            features['balconies'] = True
        if any(x in caratteristiche_str_lower for x in ['impianto tv singolo', 'impianto tv centralizzato', 'impianto tv con parabola']):
            features['tv_system'] = True
        if any(x in caratteristiche_str_lower for x in ['portiere intera giornata', 'portiere mezza giornata']):
            features['concierge'] = True
        if any(x in caratteristiche_str_lower for x in ['giardino privato', 'giardino comune', 'giardino privato e comune']):
            features['garden'] = True
        if any(x in caratteristiche_str_lower for x in ['solo cucina arredata', 'parzialmente arredato']):
            features['furnished_kitchen'] = True
        if any(x in caratteristiche_str_lower for x in ['box', 'garage', 'box/garage privato', '1 in box/garage privato']):
            features['garage_box'] = True
        if any(x in caratteristiche_str_lower for x in ['infissi', 'doppio vetro', 'triplo vetro']):
            features['high_quality_windows'] = True
        if 'security door' in caratteristiche_str_lower:
            features['security_door'] = True
        if 'terrace' in caratteristiche_str_lower:
            features['terrace'] = True
        if 'reception' in caratteristiche_str_lower:
            features['reception'] = True
        if 'jacuzzi' in caratteristiche_str_lower:
            features['jacuzzi'] = True
        if 'elevator' in caratteristiche_str_lower or 'con ascensore' in caratteristiche_str_lower:
            features['elevator'] = True
        if 'south facing' in caratteristiche_str_lower:
            features['south_facing'] = True
        if 'tavern' in caratteristiche_str_lower:
            features['tavern'] = True
        if 'video intercom' in caratteristiche_str_lower:
            features['video_intercom'] = True
        if 'swimming pool' in caratteristiche_str_lower:
            features['swimming_pool'] = True
        if 'cellar' in caratteristiche_str_lower:
            features['cellar'] = True
        if 'fireplace' in caratteristiche_str_lower:
            features['fireplace'] = True
        if 'double exposure' in caratteristiche_str_lower:
            features['double_exposure'] = True
        if 'furnished' in caratteristiche_str_lower:
            features['furnished'] = True
        if 'built-in wardrobe' in caratteristiche_str_lower:
            features['built_in_wardrobe'] = True
        if 'ground floor' in caratteristiche_str_lower:
            features['ground_floor'] = True
        if 'attic' in caratteristiche_str_lower:
            features['attic'] = True
        if 'internal exposure' in caratteristiche_str_lower:
            features['internal_exposure'] = True
        if 'electric gate' in caratteristiche_str_lower:
            features['electric_gate'] = True
        if 'external exposure' in caratteristiche_str_lower:
            features['external_exposure'] = True
        if 'disabled access' in caratteristiche_str_lower or 'con accesso disabili' in caratteristiche_str_lower:
            features['disabled_access'] = True
        if 'east facing' in caratteristiche_str_lower:
            features['east_facing'] = True
        if 'tennis court' in caratteristiche_str_lower:
            features['tennis_court'] = True
        if 'alarm system' in caratteristiche_str_lower:
            features['alarm_system'] = True
        if 'fiber optic' in caratteristiche_str_lower:
            features['fiber_optic'] = True

    return pd.Series(features)


def parse_description(row):
    """Parse description column for specific keywords."""
    description = row['description']
    features = {
        "subway": 0,
        "station": 0,
        "university": 0,
        "hospital": 0,
        "park": 0,
        "shower": 0,
        "bathtub": 0,
        "bright": 0,
        "quiet": 0,
        "wardrobe": 0,
        "market": 0,
    }

    if isinstance(description, str):
        description = description.lower()
        if "metro" or "metropolitana" in description:
            features["subway"] = 1
        if "stazione" or "treno" in description:
            features["station"] = 1
        if "universita" or "università" in description:
            features["university"] = 1
        if "ospedale" in description:
            features["hospital"] = 1
        if "parco" in description:
            features["park"] = 1
        if "doccia" in description:
            features["shower"] = 1
        if "vasca" in description:
            features["bathtub"] = 1
        if "luminoso" or "luminosa" in description:
            features["bright"] = 1
        if "silenzioso" or "silenziosa" in description:
            features["quiet"] = 1
        if "guardaroba" in description:
            features["wardrobe"] = 1
        if "mercato" or "supermercato" or "super market" or "supermarket" in description:
            features["market"] = 1

    return pd.Series(features)


def clean_deposit(row, warning=False):
    """Parse deposit (cauzione) as float."""
    deposit = row['deposit']
    if isinstance(deposit, str):
        deposit = deposit.replace('\x80', '').replace(".", "").strip()
        try:
            return float(deposit)
        except ValueError:
            if warning:
                print(f"Warning: Could not convert deposit '{row['deposit']}' to float.")
            return None
    return deposit  

municipio_df = pd.read_csv('italy-house-prices/quartieri_mapping_with_municipio.csv', encoding='latin1')

def add_municipio(row):
    """Add municipio based on neighborhood."""
    municipio_df_lower = municipio_df['Quartiere'].str.lower().str.strip()
    quartiere = row['neighborhood']
    feats = {
        'municipality': None,
        'InsideGRA': None,
        'zone': None
    }

    if isinstance(quartiere, str):
        quartiere = quartiere.lower().strip()
        match = municipio_df_lower[municipio_df_lower == quartiere]
        if not match.empty:
            municipio = municipio_df.loc[match.index[0], 'Municipio']
            inside_gra = municipio_df.loc[match.index[0], 'InsideGRA']
            zone = municipio_df.loc[match.index[0], 'Zone']
            feats['municipality'] = municipio
            feats['InsideGRA'] = inside_gra
            feats['zone'] = zone
            return pd.Series(feats)
    return None


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
    cleaned_data['price'] = data.apply(clean_price, axis=1)
    
    print("Cleaning spese condominio...")
    cleaned_data['condo_fees'] = data.apply(clean_spese_condominio, axis=1)
    
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

    print("Cleaning bathrooms...")
    bathrooms_data = data.apply(clean_bathrooms, axis=1)
    bathrooms_per_room = bathrooms_data['bathrooms'] / cleaned_data['room_total']
    bathrooms_data['bathrooms_per_room'] = bathrooms_per_room
    cleaned_data = pd.concat([cleaned_data, bathrooms_data], axis=1)
    
    print("Parsing parking...")
    temp_parking_data = data.apply(parse_parking, axis=1)
    cleaned_data = pd.concat([cleaned_data, temp_parking_data], axis=1)
    cleaned_data['has_garage_box'] = cleaned_data['garage_box'] > 0
    cleaned_data['has_outdoor_parking'] = cleaned_data['outdoor_parking'] > 0
    cleaned_data['has_common_parking'] = cleaned_data['common_parking'] > 0
    cleaned_data['has_private_box'] = cleaned_data['private_box'] > 0
    
    print("Parsing stato...")
    stato_data = data.apply(parse_stato, axis=1)
    cleaned_data = pd.concat([cleaned_data, stato_data], axis=1)
    stato_condition_dummies = pd.get_dummies(cleaned_data['stato_condition'], prefix='stato_condition')
    stato_renovation_dummies = pd.get_dummies(cleaned_data['stato_renovation'], prefix='stato_renovation')
    cleaned_data = pd.concat([cleaned_data, stato_condition_dummies, stato_renovation_dummies], axis=1)
    
    print("Parsing tipologia...")
    tipologia_data = data.apply(parse_housetype, axis=1)
    cleaned_data = pd.concat([cleaned_data, tipologia_data], axis=1)
    
    print("Cleaning disponibilita...")
    cleaned_data['is_available'] = data.apply(clean_disponibilita, axis=1)
    
    print("Cleaning anno di costruzione...")
    anno_data = data.apply(clean_anno_di_costruzione, axis=1)
    cleaned_data['construction_year'] = anno_data
    
    print("Cleaning riscaldamento...")
    riscaldamento_data = data.apply(clean_riscaldamento, axis=1)
    cleaned_data = pd.concat([cleaned_data, riscaldamento_data], axis=1)
    
    print("Cleaning air conditioning...")
    air_conditioning_data = data.apply(clean_air_conditioning, axis=1)
    cleaned_data = pd.concat([cleaned_data, air_conditioning_data], axis=1)
    
    print("Cleaning energy efficiency...")
    energy_efficiency_data = data.apply(clean_energy_efficiency, axis=1)
    cleaned_data = pd.concat([cleaned_data, energy_efficiency_data], axis=1)
    
    print("Parsing other features...")
    other_features_data = data.apply(parse_additional_features, axis=1)
    cleaned_data = pd.concat([cleaned_data, other_features_data], axis=1)

    print("Parsing description...")
    description_data = data.apply(parse_description, axis=1)
    cleaned_data = pd.concat([cleaned_data, description_data], axis=1)

    print("Cleaning deposit...")
    cleaned_data['deposit'] = data.apply(clean_deposit, axis=1)

    print("Adding municipio...")
    municipio_data = data.apply(add_municipio, axis=1)
    cleaned_data = pd.concat([cleaned_data, municipio_data], axis=1)
    
    
    # Reset index
    cleaned_data = cleaned_data.reset_index(drop=True)
    
    print(f"\nFinal cleaned dataset shape: {cleaned_data.shape}")
    
    # Save to CSV
    output_file = "italy-house-prices/rome_rents_clean_our_version.csv"
    cleaned_data.to_csv(output_file, index=False)
    print(f"\nCleaned data saved to: {output_file}")


if __name__ == "__main__":
    main()
