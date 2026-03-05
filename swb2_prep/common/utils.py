from pyproj import CRS

def crs_equal(a, b) -> bool:
    try:
        return CRS.from_user_input(a).equals(CRS.from_user_input(b))
    except Exception:
        return False