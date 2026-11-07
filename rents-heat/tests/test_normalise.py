from rents.load import building_key, normalise_building_key, strip_unit


def test_properties_and_rent_history_formats_same_key():
    a = building_key(None, "UNIT 1, 2 ARCHIBALD AVENUE WATERLOO 2017")           # /properties
    b = building_key(None, "UNIT 106, 2 ARCHIBALD AVENUE, WATERLOO NSW 2017")    # /listings/rent-history
    assert a == b == "2 ARCHIBALD AVE WATERLOO 2017"
    c = building_key(None, "UNIT 106, 18 GADIGAL AVENUE, WATERLOO NSW 2017")
    d = building_key(None, "UNIT 1, 18 GADIGAL AVENUE WATERLOO 2017")
    assert c == d == "18 GADIGAL AVE WATERLOO 2017"


def test_building_address_preferred_and_variants():
    k = building_key("18 Gadigal Avenue, Waterloo NSW 2017", "UNIT 106, 99 OTHER STREET WATERLOO 2017")
    assert k == "18 GADIGAL AVE WATERLOO 2017"
    assert normalise_building_key("18  GADIGAL AVENUE , WATERLOO (NSW) 2017") == k


def test_blank_building_address_falls_back_to_address():
    assert building_key("", "UNIT 5, 2 ARCHIBALD AVENUE WATERLOO 2017") == "2 ARCHIBALD AVE WATERLOO 2017"
    assert building_key(float("nan"), "506/20 Dunkerley Pl") == "20 DUNKERLEY PL"


def test_strip_unit_forms_and_st_peters():
    assert strip_unit("Unit G03, 5 Smith St, St Peters NSW 2044") == "5 Smith St, St Peters NSW 2044"
    assert normalise_building_key("5 Smith Street, St Peters NSW 2044") == "5 SMITH ST ST PETERS 2044"
    assert strip_unit("20 Dunkerley Pl") == "20 Dunkerley Pl"
    assert strip_unit(None) is None
