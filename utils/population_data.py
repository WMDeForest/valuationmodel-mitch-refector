import pandas as pd

countries_raw = ["Tuvalu", "Nauru", "Palau", "British Virgin Islands", "Saint Martin", "Gibraltar", "San Marino", "Monaco", "Liechtenstein", "Sint Maarten", "Marshall Islands", "American Samoa", "Turks and Caicos Islands", "Saint Kitts and Nevis", "Northern Mariana Islands", "Faroe Islands", "Greenland", "Bermuda", "Cayman Islands", "Dominica", "Andorra", "Isle of Man", "Antigua and Barbuda", "Saint Vincent and the Grenadines", "United States Virgin Islands", "Aruba", "Tonga", "Micronesia", "Seychelles", "Grenada", "Kiribati", "Curaçao", "Guam", "Saint Lucia", "Samoa", "Sao Tome and Principe", "New Caledonia", "Barbados", "French Polynesia", "Vanuatu", "Iceland", "Belize", "Bahamas", "Brunei Darussalam", "Maldives", "Malta", "Cabo Verde", "Montenegro", "Suriname", "Luxembourg", "Macao", "Solomon Islands", "Bhutan", "Guyana", "Comoros", "Fiji", "Djibouti", "Cyprus", "Mauritius", "Timor-Leste", "Estonia", "Bahrain", "Trinidad and Tobago", "Equatorial Guinea", "Kosovo", "North Macedonia", "Latvia", "Slovenia", "Guinea-Bissau", "Lesotho", "Gabon", "Moldova", "Namibia", "Botswana", "Qatar", "Albania", "Gambia", "Armenia", "Jamaica", "Lithuania", "Puerto Rico", "Bosnia and Herzegovina", "Uruguay", "Mongolia", "Eritrea", "Georgia", "Croatia", "Kuwait", "Panama", "Oman", "Mauritania", "Palestine", "Costa Rica", "New Zealand", "Ireland", "Lebanon", "Liberia", "Slovakia", "Norway", "Finland", "Central African Republic", "Singapore", "Denmark", "Congo (Republic)", "El Salvador", "Bulgaria", "Turkmenistan", "Serbia", "Paraguay", "Libya", "Nicaragua", "Kyrgyz Republic", "Hong Kong", "Lao People's Democratic Republic", "Sierra Leone", "Switzerland", "Togo", "Austria", "Belarus", "United Arab Emirates", "Hungary", "Israel", "Azerbaijan", "Tajikistan", "Papua New Guinea", "Greece", "Portugal", "Sweden", "Honduras", "Czech Republic", "South Sudan", "Cuba", "Dominican Republic", "Jordan", "Haiti", "Belgium", "Bolivia", "Tunisia", "Burundi", "Benin", "Rwanda", "Guinea", "Zimbabwe", "Cambodia", "Guatemala", "Senegal", "Netherlands", "Somalia", "Ecuador", "Chad", "Romania", "Chile", "Kazakhstan", "Zambia", "Malawi", "Sri Lanka", "Taiwan", "Syrian Arab Republic", "Burkina Faso", "Mali", "North Korea", "Australia", "Niger", "Cameroon", "Venezuela", "Cote d'Ivoire", "Madagascar", "Nepal", "Mozambique", "Ghana", "Malaysia", "Peru", "Yemen", "Uzbekistan", "Angola", "Poland", "Saudi Arabia", "Ukraine", "Morocco", "Canada", "Afghanistan", "Iraq", "Algeria", "Argentina", "Sudan", "Spain", "Uganda", "South Korea", "Colombia", "Myanmar", "Kenya", "Italy", "South Africa", "Tanzania", "France", "United Kingdom", "Thailand", "Germany", "Turkey", "Iran", "Vietnam", "Congo (Democratic)", "Egypt", "Philippines", "Japan", "Ethiopia", "Mexico", "Russian Federation", "Bangladesh", "Brazil", "Nigeria", "Pakistan", "Indonesia", "United States", "China", "India", "Swaziland", "Anguilla", "Antarctica", "Bonaire, Sint Eustatius and Saba", "Bouvet Island", "British Indian Ocean Territory", "Christmas Island", "Cocos Islands", "Cook Islands", "Falkland Islands (Malvinas)", "French Guiana", "French Southern Territories", "Guadeloupe", "Guernsey", "Heard Island and McDonald Islands", "Jersey", "Martinique", "Mayotte", "Montserrat", "Netherlands Antilles", "Niue", "Norfolk Island", "Pitcairn Islands", "Réunion", "Saint Barthélemy", "Saint Helena, Ascension and Tristan da Cunha", "Saint Pierre and Miquelon", "South Georgia and the South Sandwich Islands", "Svalbard & Jan Mayen Islands", "Tokelau", "United States Minor Outlying Islands", "Vatican City", "Wallis and Futuna", "Western Sahara", "Åland Islands"]

populations_raw = [
    11396, 12780, 18058, 31538, 32077, 32688, 33642, 36297, 39584, 41163,
    41996, 43914, 46062, 47755, 49796, 53270, 56865, 63489, 69310, 73040,
    80088, 84710, 94298, 103698, 104917, 106277, 107773, 115224, 119773,
    126183, 133515, 147862, 172952, 180251, 225681, 231856, 267940, 281995,
    308872, 334506, 393600, 410825, 412623, 452524, 521021, 553214, 598682,
    616177, 623236, 668606, 704149, 740424, 787424, 813834, 852075, 936375,
    1136455, 1260138, 1261041, 1360596, 1366188, 1485509, 1534937, 1714671,
    1756374, 1811980, 1881750, 2120937, 2150842, 2330318, 2436566, 2486891,
    2604172, 2675352, 2716391, 2745972, 2773168, 2777970, 2825544, 2871897,
    3205691, 3210847, 3423108, 3447157, 3748901, 3760365, 3853200, 4310108,
    4468087, 4644384, 4862989, 5165775, 5212173, 5223100, 5262382, 5353930,
    5418377, 5426740, 5519594, 5584264, 5742315, 5917648, 5946952, 6106869,
    6364943, 6430370, 6516100, 6618026, 6861524, 6888388, 7046310, 7100800,
    7536100, 7633779, 8791092, 8849852, 9053799, 9132383, 9178298, 9516871,
    9589872, 9756700, 10112555, 10143543, 10329931, 10361295, 10525347,
    10536632, 10593798, 10873689, 11088796, 11194449, 11332972, 11337052,
    11724763, 11822592, 12388571, 12458223, 13238559, 13712828, 14094683,
    14190612, 16665409, 16944826, 17602431, 17763163, 17879488, 18143378,
    18190484, 18278568, 19056116, 19629590, 19900177, 20569737, 20931751,
    22037000, 23000000, 23227014, 23251485, 23293698, 26160821, 26638544,
    27202843, 28647293, 28838499, 28873034, 30325732, 30896590, 33897354,
    34121985, 34308525, 34352719, 34449825, 36412350, 36684202, 36685849,
    36947025, 37000000, 37840044, 40097761, 42239854, 45504560, 45606480,
    46654581, 48109006, 48373336, 48582334, 51712619, 52085168, 54577997,
    55100586, 58761146, 60414495, 67438106, 68170228, 68350000, 71801279,
    84482267, 85326000, 89172767, 98858950, 102262808, 112716598, 117337368,
    124516650, 126527060, 128455567, 143826130, 172954319, 216422446,
    223804632, 240485658, 277534122, 334914895, 1410710000, 1428627663,
    1160000, 15000, 5000, 25000, 1, 200, 2000, 600, 17000, 3500, 300000,
    150, 400000, 63000, 1, 108000, 375000, 290000, 5000, 250000, 1600,
    2000, 50, 850000, 10000, 150000, 6000, 150, 2500, 1500, 150, 800,
    11000, 600000, 30000
]

def get_population_data():
    """
    Get the population data as a DataFrame
    
    Returns:
        pandas.DataFrame: DataFrame with country names and populations
    """
    return pd.DataFrame({'Country': countries_raw, 'Population': populations_raw}) 