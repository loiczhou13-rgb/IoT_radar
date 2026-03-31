# PlutoDoppler/config.py
class RadarConfig:
    FS = 600_000                # jusqu'à 56 MHz
    FC = 2_500_000_000            # entre 70 MHz et 6 GHz
    C = 299792458               # célérité de la lumière
    LAMBDA = C / FC
    
    BUFFER_SIZE = 65536         # nombre de points réels à stocker
    FFT_RESOLVED = 131072       # points pour FFT
    
    N_HISTORY = 120      
    WATERFALL_RES = 600 
    
    UPDATE_MS = 50       
    
    # === PARAMÈTRES DE DÉTECTION ===
    DETECTION_MARGIN = 6        # Le pic doit être 6dB au dessus du SQL pour déplacer le curseur
    V_LIMIT = 4.0        
    HZ_LIMIT = 5000      
    DB_MIN = -115        
    DB_MAX = -30