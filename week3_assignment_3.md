
### import necessary libraries


```python
import requests
from bs4 import BeautifulSoup

import pandas as pd
import geocoder

import folium

import numpy as np
from sklearn.cluster import KMeans

import matplotlib.cm as cm
import matplotlib.colors as colors
```

### obtain page as a string from the webpage


```python
quote_page='https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M'
page = requests.get(quote_page).text
```

### extract table values and organize them into a dataframe and check


```python
soup = BeautifulSoup(page, 'html.parser')
table = soup.find(lambda tag: tag.name=='table') 
table_rows = table.find_all('tr')
l = []
for tr in table_rows:
    td = tr.find_all('td')
    row = [value.text.strip('\n') for value in td]
    l.append(row)
l=l[1:]

df=pd.DataFrame(l, columns=["PostCode", "Borough", "Neighborhood"])
```

### drop rows that are not qualified and check


```python
df.drop(df[(df['Borough']=='Not assigned') | (df['Borough']==None)].index,inplace=True)
```

### find rows whose Neighborhood value is "Not assigned" and print


```python
neighborhood_not_assigned=df[df['Neighborhood']=='Not assigned']
```

### assign Borough cell value to Neighborhood cell if "Not assigned" and check


```python
df.loc[neighborhood_not_assigned.index,'Neighborhood']=neighborhood_not_assigned['Borough']
```

### merge rows sharing the same PostCode


```python
df = df.groupby('PostCode').agg({'Borough':'first', 
                             'Neighborhood': ', '.join}).reset_index()
```

### read in geospatial coordinates just in case


```python
geospatial_coordinates=pd.read_csv('Geospatial_Coordinates.csv')
```

### define a function to get coordinates given postal code
#### first try to use the geocoder method, if tried after 3 times unsuccessfully then use read in geospatial coordinates instead


```python
def get_latlng(postal_code):
    # initialize your variable to None
    lat_lng_coords = None

    # loop until you get the coordinates
    trial_count=0
    while(lat_lng_coords is None):
        g = geocoder.google('{}, Toronto, Ontario'.format(postal_code))
        lat_lng_coords = g.latlng
        trial_count+=1
        if trial_count>3:
            idx=geospatial_coordinates[geospatial_coordinates['Postal Code']==postal_code].index[0]
            lat_lng_coords=(geospatial_coordinates.iloc[idx]['Latitude'],geospatial_coordinates.iloc[idx]['Longitude'])

    return lat_lng_coords
```


```python
col_lat=[]
col_lng=[]

for row in df.iterrows():
    postal_code=row[1]['PostCode']
    lat,lng=get_latlng(postal_code)
    col_lat.append(lat)
    col_lng.append(lng)

df['Latitude']=col_lat
df['Longitude']=col_lng
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Rouge, Malvern</td>
      <td>43.806686</td>
      <td>-79.194353</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>Scarborough</td>
      <td>Highland Creek, Rouge Hill, Port Union</td>
      <td>43.784535</td>
      <td>-79.160497</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>Scarborough</td>
      <td>Guildwood, Morningside, West Hill</td>
      <td>43.763573</td>
      <td>-79.188711</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>Scarborough</td>
      <td>Woburn</td>
      <td>43.770992</td>
      <td>-79.216917</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>Scarborough</td>
      <td>Cedarbrae</td>
      <td>43.773136</td>
      <td>-79.239476</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M1J</td>
      <td>Scarborough</td>
      <td>Scarborough Village</td>
      <td>43.744734</td>
      <td>-79.239476</td>
    </tr>
    <tr>
      <th>6</th>
      <td>M1K</td>
      <td>Scarborough</td>
      <td>East Birchmount Park, Ionview, Kennedy Park</td>
      <td>43.727929</td>
      <td>-79.262029</td>
    </tr>
    <tr>
      <th>7</th>
      <td>M1L</td>
      <td>Scarborough</td>
      <td>Clairlea, Golden Mile, Oakridge</td>
      <td>43.711112</td>
      <td>-79.284577</td>
    </tr>
    <tr>
      <th>8</th>
      <td>M1M</td>
      <td>Scarborough</td>
      <td>Cliffcrest, Cliffside, Scarborough Village West</td>
      <td>43.716316</td>
      <td>-79.239476</td>
    </tr>
    <tr>
      <th>9</th>
      <td>M1N</td>
      <td>Scarborough</td>
      <td>Birch Cliff, Cliffside West</td>
      <td>43.692657</td>
      <td>-79.264848</td>
    </tr>
    <tr>
      <th>10</th>
      <td>M1P</td>
      <td>Scarborough</td>
      <td>Dorset Park, Scarborough Town Centre, Wexford ...</td>
      <td>43.757410</td>
      <td>-79.273304</td>
    </tr>
    <tr>
      <th>11</th>
      <td>M1R</td>
      <td>Scarborough</td>
      <td>Maryvale, Wexford</td>
      <td>43.750072</td>
      <td>-79.295849</td>
    </tr>
    <tr>
      <th>12</th>
      <td>M1S</td>
      <td>Scarborough</td>
      <td>Agincourt</td>
      <td>43.794200</td>
      <td>-79.262029</td>
    </tr>
    <tr>
      <th>13</th>
      <td>M1T</td>
      <td>Scarborough</td>
      <td>Clarks Corners, Sullivan, Tam O'Shanter</td>
      <td>43.781638</td>
      <td>-79.304302</td>
    </tr>
    <tr>
      <th>14</th>
      <td>M1V</td>
      <td>Scarborough</td>
      <td>Agincourt North, L'Amoreaux East, Milliken, St...</td>
      <td>43.815252</td>
      <td>-79.284577</td>
    </tr>
    <tr>
      <th>15</th>
      <td>M1W</td>
      <td>Scarborough</td>
      <td>L'Amoreaux West, Steeles West</td>
      <td>43.799525</td>
      <td>-79.318389</td>
    </tr>
    <tr>
      <th>16</th>
      <td>M1X</td>
      <td>Scarborough</td>
      <td>Upper Rouge</td>
      <td>43.836125</td>
      <td>-79.205636</td>
    </tr>
    <tr>
      <th>17</th>
      <td>M2H</td>
      <td>North York</td>
      <td>Hillcrest Village</td>
      <td>43.803762</td>
      <td>-79.363452</td>
    </tr>
    <tr>
      <th>18</th>
      <td>M2J</td>
      <td>North York</td>
      <td>Fairview, Henry Farm, Oriole</td>
      <td>43.778517</td>
      <td>-79.346556</td>
    </tr>
    <tr>
      <th>19</th>
      <td>M2K</td>
      <td>North York</td>
      <td>Bayview Village</td>
      <td>43.786947</td>
      <td>-79.385975</td>
    </tr>
    <tr>
      <th>20</th>
      <td>M2L</td>
      <td>North York</td>
      <td>Silver Hills, York Mills</td>
      <td>43.757490</td>
      <td>-79.374714</td>
    </tr>
    <tr>
      <th>21</th>
      <td>M2M</td>
      <td>North York</td>
      <td>Newtonbrook, Willowdale</td>
      <td>43.789053</td>
      <td>-79.408493</td>
    </tr>
    <tr>
      <th>22</th>
      <td>M2N</td>
      <td>North York</td>
      <td>Willowdale South</td>
      <td>43.770120</td>
      <td>-79.408493</td>
    </tr>
    <tr>
      <th>23</th>
      <td>M2P</td>
      <td>North York</td>
      <td>York Mills West</td>
      <td>43.752758</td>
      <td>-79.400049</td>
    </tr>
    <tr>
      <th>24</th>
      <td>M2R</td>
      <td>North York</td>
      <td>Willowdale West</td>
      <td>43.782736</td>
      <td>-79.442259</td>
    </tr>
    <tr>
      <th>25</th>
      <td>M3A</td>
      <td>North York</td>
      <td>Parkwoods</td>
      <td>43.753259</td>
      <td>-79.329656</td>
    </tr>
    <tr>
      <th>26</th>
      <td>M3B</td>
      <td>North York</td>
      <td>Don Mills North</td>
      <td>43.745906</td>
      <td>-79.352188</td>
    </tr>
    <tr>
      <th>27</th>
      <td>M3C</td>
      <td>North York</td>
      <td>Flemingdon Park, Don Mills South</td>
      <td>43.725900</td>
      <td>-79.340923</td>
    </tr>
    <tr>
      <th>28</th>
      <td>M3H</td>
      <td>North York</td>
      <td>Bathurst Manor, Downsview North, Wilson Heights</td>
      <td>43.754328</td>
      <td>-79.442259</td>
    </tr>
    <tr>
      <th>29</th>
      <td>M3J</td>
      <td>North York</td>
      <td>Northwood Park, York University</td>
      <td>43.767980</td>
      <td>-79.487262</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>73</th>
      <td>M6C</td>
      <td>York</td>
      <td>Humewood-Cedarvale</td>
      <td>43.693781</td>
      <td>-79.428191</td>
    </tr>
    <tr>
      <th>74</th>
      <td>M6E</td>
      <td>York</td>
      <td>Caledonia-Fairbanks</td>
      <td>43.689026</td>
      <td>-79.453512</td>
    </tr>
    <tr>
      <th>75</th>
      <td>M6G</td>
      <td>Downtown Toronto</td>
      <td>Christie</td>
      <td>43.669542</td>
      <td>-79.422564</td>
    </tr>
    <tr>
      <th>76</th>
      <td>M6H</td>
      <td>West Toronto</td>
      <td>Dovercourt Village, Dufferin</td>
      <td>43.669005</td>
      <td>-79.442259</td>
    </tr>
    <tr>
      <th>77</th>
      <td>M6J</td>
      <td>West Toronto</td>
      <td>Little Portugal, Trinity</td>
      <td>43.647927</td>
      <td>-79.419750</td>
    </tr>
    <tr>
      <th>78</th>
      <td>M6K</td>
      <td>West Toronto</td>
      <td>Brockton, Exhibition Place, Parkdale Village</td>
      <td>43.636847</td>
      <td>-79.428191</td>
    </tr>
    <tr>
      <th>79</th>
      <td>M6L</td>
      <td>North York</td>
      <td>Maple Leaf Park, North Park, Upwood Park</td>
      <td>43.713756</td>
      <td>-79.490074</td>
    </tr>
    <tr>
      <th>80</th>
      <td>M6M</td>
      <td>York</td>
      <td>Del Ray, Keelesdale, Mount Dennis, Silverthorn</td>
      <td>43.691116</td>
      <td>-79.476013</td>
    </tr>
    <tr>
      <th>81</th>
      <td>M6N</td>
      <td>York</td>
      <td>The Junction North, Runnymede</td>
      <td>43.673185</td>
      <td>-79.487262</td>
    </tr>
    <tr>
      <th>82</th>
      <td>M6P</td>
      <td>West Toronto</td>
      <td>High Park, The Junction South</td>
      <td>43.661608</td>
      <td>-79.464763</td>
    </tr>
    <tr>
      <th>83</th>
      <td>M6R</td>
      <td>West Toronto</td>
      <td>Parkdale, Roncesvalles</td>
      <td>43.648960</td>
      <td>-79.456325</td>
    </tr>
    <tr>
      <th>84</th>
      <td>M6S</td>
      <td>West Toronto</td>
      <td>Runnymede, Swansea</td>
      <td>43.651571</td>
      <td>-79.484450</td>
    </tr>
    <tr>
      <th>85</th>
      <td>M7A</td>
      <td>Queen's Park</td>
      <td>Queen's Park</td>
      <td>43.662301</td>
      <td>-79.389494</td>
    </tr>
    <tr>
      <th>86</th>
      <td>M7R</td>
      <td>Mississauga</td>
      <td>Canada Post Gateway Processing Centre</td>
      <td>43.636966</td>
      <td>-79.615819</td>
    </tr>
    <tr>
      <th>87</th>
      <td>M7Y</td>
      <td>East Toronto</td>
      <td>Business Reply Mail Processing Centre 969 Eastern</td>
      <td>43.662744</td>
      <td>-79.321558</td>
    </tr>
    <tr>
      <th>88</th>
      <td>M8V</td>
      <td>Etobicoke</td>
      <td>Humber Bay Shores, Mimico South, New Toronto</td>
      <td>43.605647</td>
      <td>-79.501321</td>
    </tr>
    <tr>
      <th>89</th>
      <td>M8W</td>
      <td>Etobicoke</td>
      <td>Alderwood, Long Branch</td>
      <td>43.602414</td>
      <td>-79.543484</td>
    </tr>
    <tr>
      <th>90</th>
      <td>M8X</td>
      <td>Etobicoke</td>
      <td>The Kingsway, Montgomery Road, Old Mill North</td>
      <td>43.653654</td>
      <td>-79.506944</td>
    </tr>
    <tr>
      <th>91</th>
      <td>M8Y</td>
      <td>Etobicoke</td>
      <td>Humber Bay, King's Mill Park, Kingsway Park So...</td>
      <td>43.636258</td>
      <td>-79.498509</td>
    </tr>
    <tr>
      <th>92</th>
      <td>M8Z</td>
      <td>Etobicoke</td>
      <td>Kingsway Park South West, Mimico NW, The Queen...</td>
      <td>43.628841</td>
      <td>-79.520999</td>
    </tr>
    <tr>
      <th>93</th>
      <td>M9A</td>
      <td>Etobicoke</td>
      <td>Islington Avenue</td>
      <td>43.667856</td>
      <td>-79.532242</td>
    </tr>
    <tr>
      <th>94</th>
      <td>M9B</td>
      <td>Etobicoke</td>
      <td>Cloverdale, Islington, Martin Grove, Princess ...</td>
      <td>43.650943</td>
      <td>-79.554724</td>
    </tr>
    <tr>
      <th>95</th>
      <td>M9C</td>
      <td>Etobicoke</td>
      <td>Bloordale Gardens, Eringate, Markland Wood, Ol...</td>
      <td>43.643515</td>
      <td>-79.577201</td>
    </tr>
    <tr>
      <th>96</th>
      <td>M9L</td>
      <td>North York</td>
      <td>Humber Summit</td>
      <td>43.756303</td>
      <td>-79.565963</td>
    </tr>
    <tr>
      <th>97</th>
      <td>M9M</td>
      <td>North York</td>
      <td>Emery, Humberlea</td>
      <td>43.724766</td>
      <td>-79.532242</td>
    </tr>
    <tr>
      <th>98</th>
      <td>M9N</td>
      <td>York</td>
      <td>Weston</td>
      <td>43.706876</td>
      <td>-79.518188</td>
    </tr>
    <tr>
      <th>99</th>
      <td>M9P</td>
      <td>Etobicoke</td>
      <td>Westmount</td>
      <td>43.696319</td>
      <td>-79.532242</td>
    </tr>
    <tr>
      <th>100</th>
      <td>M9R</td>
      <td>Etobicoke</td>
      <td>Kingsview Village, Martin Grove Gardens, Richv...</td>
      <td>43.688905</td>
      <td>-79.554724</td>
    </tr>
    <tr>
      <th>101</th>
      <td>M9V</td>
      <td>Etobicoke</td>
      <td>Albion Gardens, Beaumond Heights, Humbergate, ...</td>
      <td>43.739416</td>
      <td>-79.588437</td>
    </tr>
    <tr>
      <th>102</th>
      <td>M9W</td>
      <td>Etobicoke</td>
      <td>Northwest</td>
      <td>43.706748</td>
      <td>-79.594054</td>
    </tr>
  </tbody>
</table>
<p>103 rows × 5 columns</p>
</div>



### keep Boroughs who have "Toronto" within their names


```python
neighborhoods=df[df['Borough'].str.contains("Toronto")].reset_index(drop=True)
neighborhoods
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M4E</td>
      <td>East Toronto</td>
      <td>The Beaches</td>
      <td>43.676357</td>
      <td>-79.293031</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M4K</td>
      <td>East Toronto</td>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M4L</td>
      <td>East Toronto</td>
      <td>The Beaches West, India Bazaar</td>
      <td>43.668999</td>
      <td>-79.315572</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M4M</td>
      <td>East Toronto</td>
      <td>Studio District</td>
      <td>43.659526</td>
      <td>-79.340923</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M4N</td>
      <td>Central Toronto</td>
      <td>Lawrence Park</td>
      <td>43.728020</td>
      <td>-79.388790</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M4P</td>
      <td>Central Toronto</td>
      <td>Davisville North</td>
      <td>43.712751</td>
      <td>-79.390197</td>
    </tr>
    <tr>
      <th>6</th>
      <td>M4R</td>
      <td>Central Toronto</td>
      <td>North Toronto West</td>
      <td>43.715383</td>
      <td>-79.405678</td>
    </tr>
    <tr>
      <th>7</th>
      <td>M4S</td>
      <td>Central Toronto</td>
      <td>Davisville</td>
      <td>43.704324</td>
      <td>-79.388790</td>
    </tr>
    <tr>
      <th>8</th>
      <td>M4T</td>
      <td>Central Toronto</td>
      <td>Moore Park, Summerhill East</td>
      <td>43.689574</td>
      <td>-79.383160</td>
    </tr>
    <tr>
      <th>9</th>
      <td>M4V</td>
      <td>Central Toronto</td>
      <td>Deer Park, Forest Hill SE, Rathnelly, South Hi...</td>
      <td>43.686412</td>
      <td>-79.400049</td>
    </tr>
    <tr>
      <th>10</th>
      <td>M4W</td>
      <td>Downtown Toronto</td>
      <td>Rosedale</td>
      <td>43.679563</td>
      <td>-79.377529</td>
    </tr>
    <tr>
      <th>11</th>
      <td>M4X</td>
      <td>Downtown Toronto</td>
      <td>Cabbagetown, St. James Town</td>
      <td>43.667967</td>
      <td>-79.367675</td>
    </tr>
    <tr>
      <th>12</th>
      <td>M4Y</td>
      <td>Downtown Toronto</td>
      <td>Church and Wellesley</td>
      <td>43.665860</td>
      <td>-79.383160</td>
    </tr>
    <tr>
      <th>13</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Harbourfront, Regent Park</td>
      <td>43.654260</td>
      <td>-79.360636</td>
    </tr>
    <tr>
      <th>14</th>
      <td>M5B</td>
      <td>Downtown Toronto</td>
      <td>Ryerson, Garden District</td>
      <td>43.657162</td>
      <td>-79.378937</td>
    </tr>
    <tr>
      <th>15</th>
      <td>M5C</td>
      <td>Downtown Toronto</td>
      <td>St. James Town</td>
      <td>43.651494</td>
      <td>-79.375418</td>
    </tr>
    <tr>
      <th>16</th>
      <td>M5E</td>
      <td>Downtown Toronto</td>
      <td>Berczy Park</td>
      <td>43.644771</td>
      <td>-79.373306</td>
    </tr>
    <tr>
      <th>17</th>
      <td>M5G</td>
      <td>Downtown Toronto</td>
      <td>Central Bay Street</td>
      <td>43.657952</td>
      <td>-79.387383</td>
    </tr>
    <tr>
      <th>18</th>
      <td>M5H</td>
      <td>Downtown Toronto</td>
      <td>Adelaide, King, Richmond</td>
      <td>43.650571</td>
      <td>-79.384568</td>
    </tr>
    <tr>
      <th>19</th>
      <td>M5J</td>
      <td>Downtown Toronto</td>
      <td>Harbourfront East, Toronto Islands, Union Station</td>
      <td>43.640816</td>
      <td>-79.381752</td>
    </tr>
    <tr>
      <th>20</th>
      <td>M5K</td>
      <td>Downtown Toronto</td>
      <td>Design Exchange, Toronto Dominion Centre</td>
      <td>43.647177</td>
      <td>-79.381576</td>
    </tr>
    <tr>
      <th>21</th>
      <td>M5L</td>
      <td>Downtown Toronto</td>
      <td>Commerce Court, Victoria Hotel</td>
      <td>43.648198</td>
      <td>-79.379817</td>
    </tr>
    <tr>
      <th>22</th>
      <td>M5N</td>
      <td>Central Toronto</td>
      <td>Roselawn</td>
      <td>43.711695</td>
      <td>-79.416936</td>
    </tr>
    <tr>
      <th>23</th>
      <td>M5P</td>
      <td>Central Toronto</td>
      <td>Forest Hill North, Forest Hill West</td>
      <td>43.696948</td>
      <td>-79.411307</td>
    </tr>
    <tr>
      <th>24</th>
      <td>M5R</td>
      <td>Central Toronto</td>
      <td>The Annex, North Midtown, Yorkville</td>
      <td>43.672710</td>
      <td>-79.405678</td>
    </tr>
    <tr>
      <th>25</th>
      <td>M5S</td>
      <td>Downtown Toronto</td>
      <td>Harbord, University of Toronto</td>
      <td>43.662696</td>
      <td>-79.400049</td>
    </tr>
    <tr>
      <th>26</th>
      <td>M5T</td>
      <td>Downtown Toronto</td>
      <td>Chinatown, Grange Park, Kensington Market</td>
      <td>43.653206</td>
      <td>-79.400049</td>
    </tr>
    <tr>
      <th>27</th>
      <td>M5V</td>
      <td>Downtown Toronto</td>
      <td>CN Tower, Bathurst Quay, Island airport, Harbo...</td>
      <td>43.628947</td>
      <td>-79.394420</td>
    </tr>
    <tr>
      <th>28</th>
      <td>M5W</td>
      <td>Downtown Toronto</td>
      <td>Stn A PO Boxes 25 The Esplanade</td>
      <td>43.646435</td>
      <td>-79.374846</td>
    </tr>
    <tr>
      <th>29</th>
      <td>M5X</td>
      <td>Downtown Toronto</td>
      <td>First Canadian Place, Underground city</td>
      <td>43.648429</td>
      <td>-79.382280</td>
    </tr>
    <tr>
      <th>30</th>
      <td>M6G</td>
      <td>Downtown Toronto</td>
      <td>Christie</td>
      <td>43.669542</td>
      <td>-79.422564</td>
    </tr>
    <tr>
      <th>31</th>
      <td>M6H</td>
      <td>West Toronto</td>
      <td>Dovercourt Village, Dufferin</td>
      <td>43.669005</td>
      <td>-79.442259</td>
    </tr>
    <tr>
      <th>32</th>
      <td>M6J</td>
      <td>West Toronto</td>
      <td>Little Portugal, Trinity</td>
      <td>43.647927</td>
      <td>-79.419750</td>
    </tr>
    <tr>
      <th>33</th>
      <td>M6K</td>
      <td>West Toronto</td>
      <td>Brockton, Exhibition Place, Parkdale Village</td>
      <td>43.636847</td>
      <td>-79.428191</td>
    </tr>
    <tr>
      <th>34</th>
      <td>M6P</td>
      <td>West Toronto</td>
      <td>High Park, The Junction South</td>
      <td>43.661608</td>
      <td>-79.464763</td>
    </tr>
    <tr>
      <th>35</th>
      <td>M6R</td>
      <td>West Toronto</td>
      <td>Parkdale, Roncesvalles</td>
      <td>43.648960</td>
      <td>-79.456325</td>
    </tr>
    <tr>
      <th>36</th>
      <td>M6S</td>
      <td>West Toronto</td>
      <td>Runnymede, Swansea</td>
      <td>43.651571</td>
      <td>-79.484450</td>
    </tr>
    <tr>
      <th>37</th>
      <td>M7Y</td>
      <td>East Toronto</td>
      <td>Business Reply Mail Processing Centre 969 Eastern</td>
      <td>43.662744</td>
      <td>-79.321558</td>
    </tr>
  </tbody>
</table>
</div>



### visualize all the Toronto neighborhoods


```python
latitude,longitude=43.68, -79.39
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=11)

# add markers to map
for lat, lng, borough, neighborhood in zip(neighborhoods['Latitude'], neighborhoods['Longitude'], neighborhoods['Borough'], neighborhoods['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  
    
map_toronto
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><iframe src="data:text/html;charset=utf-8;base64,PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVMgPSBmYWxzZTsgTF9OT19UT1VDSCA9IGZhbHNlOyBMX0RJU0FCTEVfM0QgPSBmYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2FqYXguZ29vZ2xlYXBpcy5jb20vYWpheC9saWJzL2pxdWVyeS8xLjExLjEvanF1ZXJ5Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9yYXdnaXQuY29tL3B5dGhvbi12aXN1YWxpemF0aW9uL2ZvbGl1bS9tYXN0ZXIvZm9saXVtL3RlbXBsYXRlcy9sZWFmbGV0LmF3ZXNvbWUucm90YXRlLmNzcyIvPgogICAgPHN0eWxlPmh0bWwsIGJvZHkge3dpZHRoOiAxMDAlO2hlaWdodDogMTAwJTttYXJnaW46IDA7cGFkZGluZzogMDt9PC9zdHlsZT4KICAgIDxzdHlsZT4jbWFwIHtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MDtib3R0b206MDtyaWdodDowO2xlZnQ6MDt9PC9zdHlsZT4KICAgIAogICAgICAgICAgICA8c3R5bGU+ICNtYXBfY2FhNjUwNjg4NzIxNDY2MDgyOGIwM2RjNjg2MzkxNTEgewogICAgICAgICAgICAgICAgcG9zaXRpb24gOiByZWxhdGl2ZTsKICAgICAgICAgICAgICAgIHdpZHRoIDogMTAwLjAlOwogICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgICAgICAgICAgdG9wOiAwLjAlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICA8L3N0eWxlPgogICAgICAgIAo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgICAgICAgICA8ZGl2IGNsYXNzPSJmb2xpdW0tbWFwIiBpZD0ibWFwX2NhYTY1MDY4ODcyMTQ2NjA4MjhiMDNkYzY4NjM5MTUxIiA+PC9kaXY+CiAgICAgICAgCjwvYm9keT4KPHNjcmlwdD4gICAgCiAgICAKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGJvdW5kcyA9IG51bGw7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIG1hcF9jYWE2NTA2ODg3MjE0NjYwODI4YjAzZGM2ODYzOTE1MSA9IEwubWFwKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ21hcF9jYWE2NTA2ODg3MjE0NjYwODI4YjAzZGM2ODYzOTE1MScsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7Y2VudGVyOiBbNDMuNjgsLTc5LjM5XSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHpvb206IDExLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbWF4Qm91bmRzOiBib3VuZHMsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBsYXllcnM6IFtdLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgd29ybGRDb3B5SnVtcDogZmFsc2UsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBjcnM6IEwuQ1JTLkVQU0czODU3CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIH0pOwogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgdGlsZV9sYXllcl80NzBmNjY4YTI0OTU0NjI5YWE5ZDczNjQ0YmI0NmQ1NyA9IEwudGlsZUxheWVyKAogICAgICAgICAgICAgICAgJ2h0dHBzOi8ve3N9LnRpbGUub3BlbnN0cmVldG1hcC5vcmcve3p9L3t4fS97eX0ucG5nJywKICAgICAgICAgICAgICAgIHsKICAiYXR0cmlidXRpb24iOiBudWxsLAogICJkZXRlY3RSZXRpbmEiOiBmYWxzZSwKICAibWF4Wm9vbSI6IDE4LAogICJtaW5ab29tIjogMSwKICAibm9XcmFwIjogZmFsc2UsCiAgInN1YmRvbWFpbnMiOiAiYWJjIgp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jYWE2NTA2ODg3MjE0NjYwODI4YjAzZGM2ODYzOTE1MSk7CiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNDgwYjU2ZDQ0MWU1NDkzZDgyZWI5ZGQzMmViYzNhMGUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NzYzNTczOTk5OTk5OSwtNzkuMjkzMDMxMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jYWE2NTA2ODg3MjE0NjYwODI4YjAzZGM2ODYzOTE1MSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80ZTViMTFhYzgyMTg0YWQ5OTM2YmRkY2YwMjkyNjc5MSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iY2Q0NGYyNzNmMzA0ODBlYTQ0YjZiNDA2NWEzOGQ4ZiA9ICQoJzxkaXYgaWQ9Imh0bWxfYmNkNDRmMjczZjMwNDgwZWE0NGI2YjQwNjVhMzhkOGYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRoZSBCZWFjaGVzLCBFYXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzRlNWIxMWFjODIxODRhZDk5MzZiZGRjZjAyOTI2NzkxLnNldENvbnRlbnQoaHRtbF9iY2Q0NGYyNzNmMzA0ODBlYTQ0YjZiNDA2NWEzOGQ4Zik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80ODBiNTZkNDQxZTU0OTNkODJlYjlkZDMyZWJjM2EwZS5iaW5kUG9wdXAocG9wdXBfNGU1YjExYWM4MjE4NGFkOTkzNmJkZGNmMDI5MjY3OTEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZjI5MmU1YjM4Yjg3NDEzNGI3NGE0MDdhZTFiNGNkMTEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Nzk1NTcxLC03OS4zNTIxODhdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfY2FhNjUwNjg4NzIxNDY2MDgyOGIwM2RjNjg2MzkxNTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMTg5ODUwNDNjOWIyNDkxZjlkMDNmMjYyZTkyOTgxNmMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNmZlOGQzMWM5YzE5NGNkOWI4ODI3OWQyODdmYTE2YjggPSAkKCc8ZGl2IGlkPSJodG1sXzZmZThkMzFjOWMxOTRjZDliODgyNzlkMjg3ZmExNmI4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UaGUgRGFuZm9ydGggV2VzdCwgUml2ZXJkYWxlLCBFYXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzE4OTg1MDQzYzliMjQ5MWY5ZDAzZjI2MmU5Mjk4MTZjLnNldENvbnRlbnQoaHRtbF82ZmU4ZDMxYzljMTk0Y2Q5Yjg4Mjc5ZDI4N2ZhMTZiOCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mMjkyZTViMzhiODc0MTM0Yjc0YTQwN2FlMWI0Y2QxMS5iaW5kUG9wdXAocG9wdXBfMTg5ODUwNDNjOWIyNDkxZjlkMDNmMjYyZTkyOTgxNmMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZDdkZDFkNzlmYjY3NGY3Yzk1NmQ0OWMzZGUwNzQ2YTAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Njg5OTg1LC03OS4zMTU1NzE1OTk5OTk5OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jYWE2NTA2ODg3MjE0NjYwODI4YjAzZGM2ODYzOTE1MSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80ODQzN2ZkYzZjOTk0MGU2OTIwZjFhN2E1YWM4N2Y3NSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80NDA3MDZlOGQ0MzM0YTdmYTZlNDc0NzcxNjI2ZTBhZiA9ICQoJzxkaXYgaWQ9Imh0bWxfNDQwNzA2ZThkNDMzNGE3ZmE2ZTQ3NDc3MTYyNmUwYWYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRoZSBCZWFjaGVzIFdlc3QsIEluZGlhIEJhemFhciwgRWFzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80ODQzN2ZkYzZjOTk0MGU2OTIwZjFhN2E1YWM4N2Y3NS5zZXRDb250ZW50KGh0bWxfNDQwNzA2ZThkNDMzNGE3ZmE2ZTQ3NDc3MTYyNmUwYWYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZDdkZDFkNzlmYjY3NGY3Yzk1NmQ0OWMzZGUwNzQ2YTAuYmluZFBvcHVwKHBvcHVwXzQ4NDM3ZmRjNmM5OTQwZTY5MjBmMWE3YTVhYzg3Zjc1KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2FiNGExMGUwZDI0ODRiOTNiNzI5NzZhZjhiYzRiNzQxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU5NTI1NSwtNzkuMzQwOTIzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2NhYTY1MDY4ODcyMTQ2NjA4MjhiMDNkYzY4NjM5MTUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzIzMjk5OGU3ZWMzZjRmNWY5ZWE3NmVkZjlhMjliMzgwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzk1NDNlZmRjZWRiZDRiYTdiMmEzY2I4YjNmMDk0MWRhID0gJCgnPGRpdiBpZD0iaHRtbF85NTQzZWZkY2VkYmQ0YmE3YjJhM2NiOGIzZjA5NDFkYSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U3R1ZGlvIERpc3RyaWN0LCBFYXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzIzMjk5OGU3ZWMzZjRmNWY5ZWE3NmVkZjlhMjliMzgwLnNldENvbnRlbnQoaHRtbF85NTQzZWZkY2VkYmQ0YmE3YjJhM2NiOGIzZjA5NDFkYSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hYjRhMTBlMGQyNDg0YjkzYjcyOTc2YWY4YmM0Yjc0MS5iaW5kUG9wdXAocG9wdXBfMjMyOTk4ZTdlYzNmNGY1ZjllYTc2ZWRmOWEyOWIzODApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMGZkMDY0MjJmMzNjNGE0MGIwOGVlYWFhNjc0Mjg4ZTggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MjgwMjA1LC03OS4zODg3OTAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2NhYTY1MDY4ODcyMTQ2NjA4MjhiMDNkYzY4NjM5MTUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzkxOTFjZTQ5N2FkNzRiYzA5YTE5MDVjZWMwYjg2ZWUzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2Q5N2VmOTE5NjBjNzQ5MzBhMjVkYzhjYzM4YTM3MWEwID0gJCgnPGRpdiBpZD0iaHRtbF9kOTdlZjkxOTYwYzc0OTMwYTI1ZGM4Y2MzOGEzNzFhMCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TGF3cmVuY2UgUGFyaywgQ2VudHJhbCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85MTkxY2U0OTdhZDc0YmMwOWExOTA1Y2VjMGI4NmVlMy5zZXRDb250ZW50KGh0bWxfZDk3ZWY5MTk2MGM3NDkzMGEyNWRjOGNjMzhhMzcxYTApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMGZkMDY0MjJmMzNjNGE0MGIwOGVlYWFhNjc0Mjg4ZTguYmluZFBvcHVwKHBvcHVwXzkxOTFjZTQ5N2FkNzRiYzA5YTE5MDVjZWMwYjg2ZWUzKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2NjM2FhYzRkNzA5NDQ0NjU5ODdhNzY1NTQ3NzMyYjBjID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzEyNzUxMSwtNzkuMzkwMTk3NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jYWE2NTA2ODg3MjE0NjYwODI4YjAzZGM2ODYzOTE1MSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zYzJmMTc2N2M5ZGQ0NTFjYjNhZmU3NjM0NWFhMjQ3YSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hZTYyMjIzMzRiOTQ0MGEyYTk5NjJkNTQyOGZhMDFiMCA9ICQoJzxkaXYgaWQ9Imh0bWxfYWU2MjIyMzM0Yjk0NDBhMmE5OTYyZDU0MjhmYTAxYjAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRhdmlzdmlsbGUgTm9ydGgsIENlbnRyYWwgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfM2MyZjE3NjdjOWRkNDUxY2IzYWZlNzYzNDVhYTI0N2Euc2V0Q29udGVudChodG1sX2FlNjIyMjMzNGI5NDQwYTJhOTk2MmQ1NDI4ZmEwMWIwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2NjM2FhYzRkNzA5NDQ0NjU5ODdhNzY1NTQ3NzMyYjBjLmJpbmRQb3B1cChwb3B1cF8zYzJmMTc2N2M5ZGQ0NTFjYjNhZmU3NjM0NWFhMjQ3YSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81ZTlhMmE4MmY3M2Q0ZmFjOTgyMzFiNTY0N2EyNDQ2ZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcxNTM4MzQsLTc5LjQwNTY3ODQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2NhYTY1MDY4ODcyMTQ2NjA4MjhiMDNkYzY4NjM5MTUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzQwMjAzNTRlNDY3ZTQ4MjVhYTcyNTQwZTFmZjNiYzQzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2I0ZTFhZDEwZjA0NzQ2YTc5MTg4MGFiYzZmYTFiMjRjID0gJCgnPGRpdiBpZD0iaHRtbF9iNGUxYWQxMGYwNDc0NmE3OTE4ODBhYmM2ZmExYjI0YyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Tm9ydGggVG9yb250byBXZXN0LCBDZW50cmFsIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzQwMjAzNTRlNDY3ZTQ4MjVhYTcyNTQwZTFmZjNiYzQzLnNldENvbnRlbnQoaHRtbF9iNGUxYWQxMGYwNDc0NmE3OTE4ODBhYmM2ZmExYjI0Yyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81ZTlhMmE4MmY3M2Q0ZmFjOTgyMzFiNTY0N2EyNDQ2ZC5iaW5kUG9wdXAocG9wdXBfNDAyMDM1NGU0NjdlNDgyNWFhNzI1NDBlMWZmM2JjNDMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNzYwMTVmMWNkZjNlNDFmZmEwNTEyMTA2ZTVkMDI1ZTUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MDQzMjQ0LC03OS4zODg3OTAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2NhYTY1MDY4ODcyMTQ2NjA4MjhiMDNkYzY4NjM5MTUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzRhOTZmZjlkY2QwZjRlMTc4ODEzNTJmMzVhZmZiN2VhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzlkZjk5ZDdmNjNjYjRkYzI5NjViMWM1ZGVkYjcxNTYzID0gJCgnPGRpdiBpZD0iaHRtbF85ZGY5OWQ3ZjYzY2I0ZGMyOTY1YjFjNWRlZGI3MTU2MyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RGF2aXN2aWxsZSwgQ2VudHJhbCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80YTk2ZmY5ZGNkMGY0ZTE3ODgxMzUyZjM1YWZmYjdlYS5zZXRDb250ZW50KGh0bWxfOWRmOTlkN2Y2M2NiNGRjMjk2NWIxYzVkZWRiNzE1NjMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNzYwMTVmMWNkZjNlNDFmZmEwNTEyMTA2ZTVkMDI1ZTUuYmluZFBvcHVwKHBvcHVwXzRhOTZmZjlkY2QwZjRlMTc4ODEzNTJmMzVhZmZiN2VhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzliMDc5MDBjMjNlNTQ4MzU4NTUyMGExYWFiZDBkYTM0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjg5NTc0MywtNzkuMzgzMTU5OTAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfY2FhNjUwNjg4NzIxNDY2MDgyOGIwM2RjNjg2MzkxNTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNzQwMDA0N2VlZjY1NDMwM2JkNjFjMjg2ZDFhZjg4OGEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNGYxMjE4NTYwYjIxNDY1OTk0ODNjZTllNmUyOTQ3YWMgPSAkKCc8ZGl2IGlkPSJodG1sXzRmMTIxODU2MGIyMTQ2NTk5NDgzY2U5ZTZlMjk0N2FjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Nb29yZSBQYXJrLCBTdW1tZXJoaWxsIEVhc3QsIENlbnRyYWwgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNzQwMDA0N2VlZjY1NDMwM2JkNjFjMjg2ZDFhZjg4OGEuc2V0Q29udGVudChodG1sXzRmMTIxODU2MGIyMTQ2NTk5NDgzY2U5ZTZlMjk0N2FjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzliMDc5MDBjMjNlNTQ4MzU4NTUyMGExYWFiZDBkYTM0LmJpbmRQb3B1cChwb3B1cF83NDAwMDQ3ZWVmNjU0MzAzYmQ2MWMyODZkMWFmODg4YSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85ZWEzNDUyMWJjZDE0MjZmOGM0NGI2OGMwOTI4M2JhZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY4NjQxMjI5OTk5OTk5LC03OS40MDAwNDkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2NhYTY1MDY4ODcyMTQ2NjA4MjhiMDNkYzY4NjM5MTUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzYxMGNmNDRhM2FhMjQwYjk4MDgxZTA1ZmVlMTQyZDdmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2YzYTViNDAzNDk3OTQ1Yzk5OTRmZjgxMjk4MmYxMWY3ID0gJCgnPGRpdiBpZD0iaHRtbF9mM2E1YjQwMzQ5Nzk0NWM5OTk0ZmY4MTI5ODJmMTFmNyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RGVlciBQYXJrLCBGb3Jlc3QgSGlsbCBTRSwgUmF0aG5lbGx5LCBTb3V0aCBIaWxsLCBTdW1tZXJoaWxsIFdlc3QsIENlbnRyYWwgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNjEwY2Y0NGEzYWEyNDBiOTgwODFlMDVmZWUxNDJkN2Yuc2V0Q29udGVudChodG1sX2YzYTViNDAzNDk3OTQ1Yzk5OTRmZjgxMjk4MmYxMWY3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzllYTM0NTIxYmNkMTQyNmY4YzQ0YjY4YzA5MjgzYmFkLmJpbmRQb3B1cChwb3B1cF82MTBjZjQ0YTNhYTI0MGI5ODA4MWUwNWZlZTE0MmQ3Zik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mNzc1NmI0MzMwMjc0NzYyODIyMWYxOGEzZTJmYjM2OSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY3OTU2MjYsLTc5LjM3NzUyOTQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2NhYTY1MDY4ODcyMTQ2NjA4MjhiMDNkYzY4NjM5MTUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzQ0NDgyNmRhNDRmNDRlNjhhMTNmYzhkYjcwOWMwYjJkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzY0YjdjODk5ZTkzOTQ3MjRhZDEwZGFkMzliMjg2NjhkID0gJCgnPGRpdiBpZD0iaHRtbF82NGI3Yzg5OWU5Mzk0NzI0YWQxMGRhZDM5YjI4NjY4ZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Um9zZWRhbGUsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzQ0NDgyNmRhNDRmNDRlNjhhMTNmYzhkYjcwOWMwYjJkLnNldENvbnRlbnQoaHRtbF82NGI3Yzg5OWU5Mzk0NzI0YWQxMGRhZDM5YjI4NjY4ZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mNzc1NmI0MzMwMjc0NzYyODIyMWYxOGEzZTJmYjM2OS5iaW5kUG9wdXAocG9wdXBfNDQ0ODI2ZGE0NGY0NGU2OGExM2ZjOGRiNzA5YzBiMmQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYjU4ZDA1ZmJjYzA1NDhhNzhhNjhlODBkM2JkMDI5MjQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Njc5NjcsLTc5LjM2NzY3NTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfY2FhNjUwNjg4NzIxNDY2MDgyOGIwM2RjNjg2MzkxNTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYzgzZGYwMmZiNGQ3NDkxNzk4YjIxOTdiYTE4MmQ0ZjcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZDhlOWU1YjcyMDViNGRmYWE2OGE5ODU1Yzc4M2FhMzggPSAkKCc8ZGl2IGlkPSJodG1sX2Q4ZTllNWI3MjA1YjRkZmFhNjhhOTg1NWM3ODNhYTM4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DYWJiYWdldG93biwgU3QuIEphbWVzIFRvd24sIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2M4M2RmMDJmYjRkNzQ5MTc5OGIyMTk3YmExODJkNGY3LnNldENvbnRlbnQoaHRtbF9kOGU5ZTViNzIwNWI0ZGZhYTY4YTk4NTVjNzgzYWEzOCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iNThkMDVmYmNjMDU0OGE3OGE2OGU4MGQzYmQwMjkyNC5iaW5kUG9wdXAocG9wdXBfYzgzZGYwMmZiNGQ3NDkxNzk4YjIxOTdiYTE4MmQ0ZjcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMmFiZWI0ZTE1NDlhNDhjYmEwMzUzODYzMTJiOGIyMTUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjU4NTk5LC03OS4zODMxNTk5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jYWE2NTA2ODg3MjE0NjYwODI4YjAzZGM2ODYzOTE1MSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yNzI5YTEwOTM0YmU0NmZmYjZlZjJjYWFkOTUyOTVkYSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xYTEyZjc0ZDdiMzM0N2Q4YjI3OGIwZTJjMjM0NzA1NSA9ICQoJzxkaXYgaWQ9Imh0bWxfMWExMmY3NGQ3YjMzNDdkOGIyNzhiMGUyYzIzNDcwNTUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNodXJjaCBhbmQgV2VsbGVzbGV5LCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yNzI5YTEwOTM0YmU0NmZmYjZlZjJjYWFkOTUyOTVkYS5zZXRDb250ZW50KGh0bWxfMWExMmY3NGQ3YjMzNDdkOGIyNzhiMGUyYzIzNDcwNTUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMmFiZWI0ZTE1NDlhNDhjYmEwMzUzODYzMTJiOGIyMTUuYmluZFBvcHVwKHBvcHVwXzI3MjlhMTA5MzRiZTQ2ZmZiNmVmMmNhYWQ5NTI5NWRhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzJkMDJmYTViMjYxZDQ3NGQ5ZmZjNDAwZmVlYTNiYTBiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU0MjU5OSwtNzkuMzYwNjM1OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jYWE2NTA2ODg3MjE0NjYwODI4YjAzZGM2ODYzOTE1MSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hYzU4MjMxMTFmZGE0MzMzYjdiNGYyYmFiMTg3M2Y4MSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jNDcyZmY1YmQ2YjA0N2MxODNkZTUxZjZkNmRhMmNiZiA9ICQoJzxkaXYgaWQ9Imh0bWxfYzQ3MmZmNWJkNmIwNDdjMTgzZGU1MWY2ZDZkYTJjYmYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkhhcmJvdXJmcm9udCwgUmVnZW50IFBhcmssIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2FjNTgyMzExMWZkYTQzMzNiN2I0ZjJiYWIxODczZjgxLnNldENvbnRlbnQoaHRtbF9jNDcyZmY1YmQ2YjA0N2MxODNkZTUxZjZkNmRhMmNiZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yZDAyZmE1YjI2MWQ0NzRkOWZmYzQwMGZlZWEzYmEwYi5iaW5kUG9wdXAocG9wdXBfYWM1ODIzMTExZmRhNDMzM2I3YjRmMmJhYjE4NzNmODEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZjEwY2UxOTQwYmYyNDNhOTk1ODFhMjQ2NGJkZmZhOTkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTcxNjE4LC03OS4zNzg5MzcwOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jYWE2NTA2ODg3MjE0NjYwODI4YjAzZGM2ODYzOTE1MSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xOTI3NWQxMTEyOTg0ZmEwOWU2ZjExMmZlNDBmM2Q1NSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yNzI5NjViOTc0YWE0OTVkOGVhNWM4OWI4ZTA1ZDgzMyA9ICQoJzxkaXYgaWQ9Imh0bWxfMjcyOTY1Yjk3NGFhNDk1ZDhlYTVjODliOGUwNWQ4MzMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJ5ZXJzb24sIEdhcmRlbiBEaXN0cmljdCwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMTkyNzVkMTExMjk4NGZhMDllNmYxMTJmZTQwZjNkNTUuc2V0Q29udGVudChodG1sXzI3Mjk2NWI5NzRhYTQ5NWQ4ZWE1Yzg5YjhlMDVkODMzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2YxMGNlMTk0MGJmMjQzYTk5NTgxYTI0NjRiZGZmYTk5LmJpbmRQb3B1cChwb3B1cF8xOTI3NWQxMTEyOTg0ZmEwOWU2ZjExMmZlNDBmM2Q1NSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zNzE0NDlhNGFkMjU0MGFjOTYxMzNjMDBhNmE4OTY4MiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MTQ5MzksLTc5LjM3NTQxNzldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfY2FhNjUwNjg4NzIxNDY2MDgyOGIwM2RjNjg2MzkxNTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMWY2Yzk3ODRhMWVhNDIyMzk3NjhiOGMxNTA2MWUxZWYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfY2YxNmUxMDVhNmVjNDRjMTg4ZjM3ZmU2ZTc3MWE0MDggPSAkKCc8ZGl2IGlkPSJodG1sX2NmMTZlMTA1YTZlYzQ0YzE4OGYzN2ZlNmU3NzFhNDA4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TdC4gSmFtZXMgVG93biwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMWY2Yzk3ODRhMWVhNDIyMzk3NjhiOGMxNTA2MWUxZWYuc2V0Q29udGVudChodG1sX2NmMTZlMTA1YTZlYzQ0YzE4OGYzN2ZlNmU3NzFhNDA4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzM3MTQ0OWE0YWQyNTQwYWM5NjEzM2MwMGE2YTg5NjgyLmJpbmRQb3B1cChwb3B1cF8xZjZjOTc4NGExZWE0MjIzOTc2OGI4YzE1MDYxZTFlZik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kN2JmMGZiOWYxZWI0OTQwYmU2YjE2YWRjZWQ3ZDg2MSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NDc3MDc5OTk5OTk5NiwtNzkuMzczMzA2NF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jYWE2NTA2ODg3MjE0NjYwODI4YjAzZGM2ODYzOTE1MSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mMWM1Y2ZkNTc2NTM0NDAxODlhZjI5NDQ0YTc4ZTdiZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zNWE1NmE0NzczMTg0ZGNmOTY4MmQ2ODc0ZWYwYmRlNyA9ICQoJzxkaXYgaWQ9Imh0bWxfMzVhNTZhNDc3MzE4NGRjZjk2ODJkNjg3NGVmMGJkZTciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJlcmN6eSBQYXJrLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mMWM1Y2ZkNTc2NTM0NDAxODlhZjI5NDQ0YTc4ZTdiZi5zZXRDb250ZW50KGh0bWxfMzVhNTZhNDc3MzE4NGRjZjk2ODJkNjg3NGVmMGJkZTcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZDdiZjBmYjlmMWViNDk0MGJlNmIxNmFkY2VkN2Q4NjEuYmluZFBvcHVwKHBvcHVwX2YxYzVjZmQ1NzY1MzQ0MDE4OWFmMjk0NDRhNzhlN2JmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzgyYTdlYjk0MjRmYzQ0ZjhhNzNjYTYwYTYxOTk2NTZkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU3OTUyNCwtNzkuMzg3MzgyNl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jYWE2NTA2ODg3MjE0NjYwODI4YjAzZGM2ODYzOTE1MSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8wY2M5NjIwODkyNjM0MWU1YTY5ZDljNjliZWM5NTdiOCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kZTQxYzc0MzA2OWU0YjI5YmI5ZTNiMjEzYjJkNTFiNSA9ICQoJzxkaXYgaWQ9Imh0bWxfZGU0MWM3NDMwNjllNGIyOWJiOWUzYjIxM2IyZDUxYjUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNlbnRyYWwgQmF5IFN0cmVldCwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMGNjOTYyMDg5MjYzNDFlNWE2OWQ5YzY5YmVjOTU3Yjguc2V0Q29udGVudChodG1sX2RlNDFjNzQzMDY5ZTRiMjliYjllM2IyMTNiMmQ1MWI1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzgyYTdlYjk0MjRmYzQ0ZjhhNzNjYTYwYTYxOTk2NTZkLmJpbmRQb3B1cChwb3B1cF8wY2M5NjIwODkyNjM0MWU1YTY5ZDljNjliZWM5NTdiOCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82NWEwMjBmYTI5OTU0ZTgzODBiMDQ4NTlhYzliNzEzOCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MDU3MTIwMDAwMDAxLC03OS4zODQ1Njc1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2NhYTY1MDY4ODcyMTQ2NjA4MjhiMDNkYzY4NjM5MTUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzBiYjE1MTdkNDBlNjQxZDdhZDVhYmE1ZTRkODRiM2EzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzVmNzY4NmYzYmZkYzQ0OWVhMzA0YjQ5NzcxMzBiM2NiID0gJCgnPGRpdiBpZD0iaHRtbF81Zjc2ODZmM2JmZGM0NDllYTMwNGI0OTc3MTMwYjNjYiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QWRlbGFpZGUsIEtpbmcsIFJpY2htb25kLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wYmIxNTE3ZDQwZTY0MWQ3YWQ1YWJhNWU0ZDg0YjNhMy5zZXRDb250ZW50KGh0bWxfNWY3Njg2ZjNiZmRjNDQ5ZWEzMDRiNDk3NzEzMGIzY2IpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNjVhMDIwZmEyOTk1NGU4MzgwYjA0ODU5YWM5YjcxMzguYmluZFBvcHVwKHBvcHVwXzBiYjE1MTdkNDBlNjQxZDdhZDVhYmE1ZTRkODRiM2EzKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzAzODcwYjY1NDg2NjQyZGRiZDk0OWFkNjc2MzNhODk5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQwODE1NywtNzkuMzgxNzUyMjk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfY2FhNjUwNjg4NzIxNDY2MDgyOGIwM2RjNjg2MzkxNTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNTc1YWFmN2E1NWE0NDU2MGEyZjUxZWEyYTQzMDM1M2UgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMTNkMjMxZWI1NjYxNDJjMThlYmY1MTMxYTY5MTY0ZjggPSAkKCc8ZGl2IGlkPSJodG1sXzEzZDIzMWViNTY2MTQyYzE4ZWJmNTEzMWE2OTE2NGY4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IYXJib3VyZnJvbnQgRWFzdCwgVG9yb250byBJc2xhbmRzLCBVbmlvbiBTdGF0aW9uLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81NzVhYWY3YTU1YTQ0NTYwYTJmNTFlYTJhNDMwMzUzZS5zZXRDb250ZW50KGh0bWxfMTNkMjMxZWI1NjYxNDJjMThlYmY1MTMxYTY5MTY0ZjgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMDM4NzBiNjU0ODY2NDJkZGJkOTQ5YWQ2NzYzM2E4OTkuYmluZFBvcHVwKHBvcHVwXzU3NWFhZjdhNTVhNDQ1NjBhMmY1MWVhMmE0MzAzNTNlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2ZiNTgxNDc1NmJhNjRkNWJhZDk1MjZhNTc4YWM0MTkzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ3MTc2OCwtNzkuMzgxNTc2NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfY2FhNjUwNjg4NzIxNDY2MDgyOGIwM2RjNjg2MzkxNTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfY2Q3YmJlYWQwNzFkNGUwMDhlZTZhZWYyYjQxNjRjMzMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMGZlYjRkNGZjYzdlNGQzM2E3ZmMwM2QwZDVkMjY4ODEgPSAkKCc8ZGl2IGlkPSJodG1sXzBmZWI0ZDRmY2M3ZTRkMzNhN2ZjMDNkMGQ1ZDI2ODgxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5EZXNpZ24gRXhjaGFuZ2UsIFRvcm9udG8gRG9taW5pb24gQ2VudHJlLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jZDdiYmVhZDA3MWQ0ZTAwOGVlNmFlZjJiNDE2NGMzMy5zZXRDb250ZW50KGh0bWxfMGZlYjRkNGZjYzdlNGQzM2E3ZmMwM2QwZDVkMjY4ODEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZmI1ODE0NzU2YmE2NGQ1YmFkOTUyNmE1NzhhYzQxOTMuYmluZFBvcHVwKHBvcHVwX2NkN2JiZWFkMDcxZDRlMDA4ZWU2YWVmMmI0MTY0YzMzKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQyZmMzZjkwZDQ3MjQwZWQ5ODQ2ZjcyMmE2OGQ5YmM4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ4MTk4NSwtNzkuMzc5ODE2OTAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfY2FhNjUwNjg4NzIxNDY2MDgyOGIwM2RjNjg2MzkxNTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMjliODUxNWZjM2Y2NDQ3ZDk4MWExMjBhOTIyN2Q3MzMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOTRkYWM0N2RiMDYwNGQ0NzhjYzc1YWY3NmM1NDlmNjggPSAkKCc8ZGl2IGlkPSJodG1sXzk0ZGFjNDdkYjA2MDRkNDc4Y2M3NWFmNzZjNTQ5ZjY4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Db21tZXJjZSBDb3VydCwgVmljdG9yaWEgSG90ZWwsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzI5Yjg1MTVmYzNmNjQ0N2Q5ODFhMTIwYTkyMjdkNzMzLnNldENvbnRlbnQoaHRtbF85NGRhYzQ3ZGIwNjA0ZDQ3OGNjNzVhZjc2YzU0OWY2OCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80MmZjM2Y5MGQ0NzI0MGVkOTg0NmY3MjJhNjhkOWJjOC5iaW5kUG9wdXAocG9wdXBfMjliODUxNWZjM2Y2NDQ3ZDk4MWExMjBhOTIyN2Q3MzMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZDg3MGE2Yjc5MzhhNDc2YzlkZjQ1YzQ3MmM1M2NkMTIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MTE2OTQ4LC03OS40MTY5MzU1OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jYWE2NTA2ODg3MjE0NjYwODI4YjAzZGM2ODYzOTE1MSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mODRlZWVmMzFiZWU0ZTY2OTYzYzkzZTY4YWI5OTNkYiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84MDIyNDRhYjU4NmQ0ZGM1YTdlY2U2NWMyZjQ1MmEzNCA9ICQoJzxkaXYgaWQ9Imh0bWxfODAyMjQ0YWI1ODZkNGRjNWE3ZWNlNjVjMmY0NTJhMzQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJvc2VsYXduLCBDZW50cmFsIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Y4NGVlZWYzMWJlZTRlNjY5NjNjOTNlNjhhYjk5M2RiLnNldENvbnRlbnQoaHRtbF84MDIyNDRhYjU4NmQ0ZGM1YTdlY2U2NWMyZjQ1MmEzNCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kODcwYTZiNzkzOGE0NzZjOWRmNDVjNDcyYzUzY2QxMi5iaW5kUG9wdXAocG9wdXBfZjg0ZWVlZjMxYmVlNGU2Njk2M2M5M2U2OGFiOTkzZGIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMjBiNDAyMGUxNzkwNGM3OWE4NDU0ZGNmYTY4YjE2YTEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42OTY5NDc2LC03OS40MTEzMDcyMDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jYWE2NTA2ODg3MjE0NjYwODI4YjAzZGM2ODYzOTE1MSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mNGYwZWM3MDQzYjg0MDY0YjY2NDcxZjkwMjBmZTE5YiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mNzUwZjQyZTQzZGY0OTI5YjYxNjJiZDYzNDUzYzExNiA9ICQoJzxkaXYgaWQ9Imh0bWxfZjc1MGY0MmU0M2RmNDkyOWI2MTYyYmQ2MzQ1M2MxMTYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkZvcmVzdCBIaWxsIE5vcnRoLCBGb3Jlc3QgSGlsbCBXZXN0LCBDZW50cmFsIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Y0ZjBlYzcwNDNiODQwNjRiNjY0NzFmOTAyMGZlMTliLnNldENvbnRlbnQoaHRtbF9mNzUwZjQyZTQzZGY0OTI5YjYxNjJiZDYzNDUzYzExNik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yMGI0MDIwZTE3OTA0Yzc5YTg0NTRkY2ZhNjhiMTZhMS5iaW5kUG9wdXAocG9wdXBfZjRmMGVjNzA0M2I4NDA2NGI2NjQ3MWY5MDIwZmUxOWIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNjZkZDc3MzI5NjcyNDY5YjlhMGM5NDk3YWVjNWM3YTggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NzI3MDk3LC03OS40MDU2Nzg0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jYWE2NTA2ODg3MjE0NjYwODI4YjAzZGM2ODYzOTE1MSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hYzhkNzAwZjRiN2U0ZDg0ODk2ZjZhMDg2ZmZjNzRjMCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83MzgwNWRlZDA0OTE0OGEzOGMwZmU1MmI4YTVhYzFlYyA9ICQoJzxkaXYgaWQ9Imh0bWxfNzM4MDVkZWQwNDkxNDhhMzhjMGZlNTJiOGE1YWMxZWMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRoZSBBbm5leCwgTm9ydGggTWlkdG93biwgWW9ya3ZpbGxlLCBDZW50cmFsIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2FjOGQ3MDBmNGI3ZTRkODQ4OTZmNmEwODZmZmM3NGMwLnNldENvbnRlbnQoaHRtbF83MzgwNWRlZDA0OTE0OGEzOGMwZmU1MmI4YTVhYzFlYyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82NmRkNzczMjk2NzI0NjliOWEwYzk0OTdhZWM1YzdhOC5iaW5kUG9wdXAocG9wdXBfYWM4ZDcwMGY0YjdlNGQ4NDg5NmY2YTA4NmZmYzc0YzApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNzg3ZDU1M2UzZjlhNGVmMWI5OTE5ZjEwNGFmNTQ3NmQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjI2OTU2LC03OS40MDAwNDkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2NhYTY1MDY4ODcyMTQ2NjA4MjhiMDNkYzY4NjM5MTUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2EwMzFmZjUyNzQwMjQyNzU5YWE1ZmIwMDM5YWZmYjI5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzMwOWY5NWYzN2YyZDRkZmY4ZWFlMDAxZTVjM2Q4NmZkID0gJCgnPGRpdiBpZD0iaHRtbF8zMDlmOTVmMzdmMmQ0ZGZmOGVhZTAwMWU1YzNkODZmZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SGFyYm9yZCwgVW5pdmVyc2l0eSBvZiBUb3JvbnRvLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hMDMxZmY1Mjc0MDI0Mjc1OWFhNWZiMDAzOWFmZmIyOS5zZXRDb250ZW50KGh0bWxfMzA5Zjk1ZjM3ZjJkNGRmZjhlYWUwMDFlNWMzZDg2ZmQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNzg3ZDU1M2UzZjlhNGVmMWI5OTE5ZjEwNGFmNTQ3NmQuYmluZFBvcHVwKHBvcHVwX2EwMzFmZjUyNzQwMjQyNzU5YWE1ZmIwMDM5YWZmYjI5KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzE1ZTljZmQxOTBlYTQ2NDA5NzJlNmE5OTc2ZjBjYTYwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUzMjA1NywtNzkuNDAwMDQ5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jYWE2NTA2ODg3MjE0NjYwODI4YjAzZGM2ODYzOTE1MSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85ODlmYWRjMWQ3YTY0ZDBkOTUzNjMxZTU5MGFkNTNhNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kYzc1MzNhNjViOTg0OWQ0OThlY2NmNDQ3ZWU5MzY0ZCA9ICQoJzxkaXYgaWQ9Imh0bWxfZGM3NTMzYTY1Yjk4NDlkNDk4ZWNjZjQ0N2VlOTM2NGQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNoaW5hdG93biwgR3JhbmdlIFBhcmssIEtlbnNpbmd0b24gTWFya2V0LCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85ODlmYWRjMWQ3YTY0ZDBkOTUzNjMxZTU5MGFkNTNhNi5zZXRDb250ZW50KGh0bWxfZGM3NTMzYTY1Yjk4NDlkNDk4ZWNjZjQ0N2VlOTM2NGQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMTVlOWNmZDE5MGVhNDY0MDk3MmU2YTk5NzZmMGNhNjAuYmluZFBvcHVwKHBvcHVwXzk4OWZhZGMxZDdhNjRkMGQ5NTM2MzFlNTkwYWQ1M2E2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQyNzU1YTA0MDg1NTQ3OTZiZTU4MjIxNWQ5MWVhZjRlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjI4OTQ2NywtNzkuMzk0NDE5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jYWE2NTA2ODg3MjE0NjYwODI4YjAzZGM2ODYzOTE1MSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mMDNhYzBkOWQ3YTY0NGQ4OWM5MjVhOTM5YzRlOGE1MCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84MTZmOGQxNTVjZjU0MDQ3ODIzNTY5MTQ3NzBjMTM4ZCA9ICQoJzxkaXYgaWQ9Imh0bWxfODE2ZjhkMTU1Y2Y1NDA0NzgyMzU2OTE0NzcwYzEzOGQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNOIFRvd2VyLCBCYXRodXJzdCBRdWF5LCBJc2xhbmQgYWlycG9ydCwgSGFyYm91cmZyb250IFdlc3QsIEtpbmcgYW5kIFNwYWRpbmEsIFJhaWx3YXkgTGFuZHMsIFNvdXRoIE5pYWdhcmEsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2YwM2FjMGQ5ZDdhNjQ0ZDg5YzkyNWE5MzljNGU4YTUwLnNldENvbnRlbnQoaHRtbF84MTZmOGQxNTVjZjU0MDQ3ODIzNTY5MTQ3NzBjMTM4ZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80Mjc1NWEwNDA4NTU0Nzk2YmU1ODIyMTVkOTFlYWY0ZS5iaW5kUG9wdXAocG9wdXBfZjAzYWMwZDlkN2E2NDRkODljOTI1YTkzOWM0ZThhNTApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfY2UwNzU4ZTE2OWE3NGU5ZWFmYTBjNzdhMDJmZjA4MDUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDY0MzUyLC03OS4zNzQ4NDU5OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jYWE2NTA2ODg3MjE0NjYwODI4YjAzZGM2ODYzOTE1MSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lMDA2MWEwZWFmNjU0NWI0Yjc5YjIwMzZhYjU3MTE2MSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84OGY0NDYyNjBiZDk0NWI3ODUyY2E3ZDE5NmNmYTgzZCA9ICQoJzxkaXYgaWQ9Imh0bWxfODhmNDQ2MjYwYmQ5NDViNzg1MmNhN2QxOTZjZmE4M2QiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0biBBIFBPIEJveGVzIDI1IFRoZSBFc3BsYW5hZGUsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2UwMDYxYTBlYWY2NTQ1YjRiNzliMjAzNmFiNTcxMTYxLnNldENvbnRlbnQoaHRtbF84OGY0NDYyNjBiZDk0NWI3ODUyY2E3ZDE5NmNmYTgzZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jZTA3NThlMTY5YTc0ZTllYWZhMGM3N2EwMmZmMDgwNS5iaW5kUG9wdXAocG9wdXBfZTAwNjFhMGVhZjY1NDViNGI3OWIyMDM2YWI1NzExNjEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzY5ZDIxMjRmNzhmNDJmMGJkMTgyNTA3MDJkMWM5NjIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDg0MjkyLC03OS4zODIyODAyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2NhYTY1MDY4ODcyMTQ2NjA4MjhiMDNkYzY4NjM5MTUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2RiODI5OTJmYjFmNzRjNjFiZTdhNDBkYmU4OWFkYjU2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzEyMjRjNmVlNDc1YTRmZDhhYTI1NTkzYTg3N2JmM2MwID0gJCgnPGRpdiBpZD0iaHRtbF8xMjI0YzZlZTQ3NWE0ZmQ4YWEyNTU5M2E4NzdiZjNjMCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Rmlyc3QgQ2FuYWRpYW4gUGxhY2UsIFVuZGVyZ3JvdW5kIGNpdHksIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2RiODI5OTJmYjFmNzRjNjFiZTdhNDBkYmU4OWFkYjU2LnNldENvbnRlbnQoaHRtbF8xMjI0YzZlZTQ3NWE0ZmQ4YWEyNTU5M2E4NzdiZjNjMCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jNjlkMjEyNGY3OGY0MmYwYmQxODI1MDcwMmQxYzk2Mi5iaW5kUG9wdXAocG9wdXBfZGI4Mjk5MmZiMWY3NGM2MWJlN2E0MGRiZTg5YWRiNTYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfODQwNDlkNzM4MjIwNDc1ZWFiYWY5MmIzMDc4MmIyMDIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Njk1NDIsLTc5LjQyMjU2MzddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfY2FhNjUwNjg4NzIxNDY2MDgyOGIwM2RjNjg2MzkxNTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZDE5ZWUzYzIwMzM4NDI3MWE2YjY5ZjAxM2FmZmJmZDYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfM2RlYTI1YjFjN2NlNGRhOGE0MDVlOTM0NDc4ZTdlM2MgPSAkKCc8ZGl2IGlkPSJodG1sXzNkZWEyNWIxYzdjZTRkYThhNDA1ZTkzNDQ3OGU3ZTNjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DaHJpc3RpZSwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZDE5ZWUzYzIwMzM4NDI3MWE2YjY5ZjAxM2FmZmJmZDYuc2V0Q29udGVudChodG1sXzNkZWEyNWIxYzdjZTRkYThhNDA1ZTkzNDQ3OGU3ZTNjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzg0MDQ5ZDczODIyMDQ3NWVhYmFmOTJiMzA3ODJiMjAyLmJpbmRQb3B1cChwb3B1cF9kMTllZTNjMjAzMzg0MjcxYTZiNjlmMDEzYWZmYmZkNik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yNzBhZmQ3YjVhZTg0OWIzYjBjNWVjYjExN2UwZmVlNSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2OTAwNTEwMDAwMDAxLC03OS40NDIyNTkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2NhYTY1MDY4ODcyMTQ2NjA4MjhiMDNkYzY4NjM5MTUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzkxMWI5ZDA5ZjEzNzQ4MDliYmY4ZGFhYjIxMGRjN2JlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2ZhODExOWVlZWE3YjQ0ZjJiYjYzZmJjODkwMGFjMWM3ID0gJCgnPGRpdiBpZD0iaHRtbF9mYTgxMTllZWVhN2I0NGYyYmI2M2ZiYzg5MDBhYzFjNyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG92ZXJjb3VydCBWaWxsYWdlLCBEdWZmZXJpbiwgV2VzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85MTFiOWQwOWYxMzc0ODA5YmJmOGRhYWIyMTBkYzdiZS5zZXRDb250ZW50KGh0bWxfZmE4MTE5ZWVlYTdiNDRmMmJiNjNmYmM4OTAwYWMxYzcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMjcwYWZkN2I1YWU4NDliM2IwYzVlY2IxMTdlMGZlZTUuYmluZFBvcHVwKHBvcHVwXzkxMWI5ZDA5ZjEzNzQ4MDliYmY4ZGFhYjIxMGRjN2JlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzliMTI1MzA5MTQ0ZjQyNmVhYzIyYWY3MTE1MDJiZWRiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ3OTI2NzAwMDAwMDA2LC03OS40MTk3NDk3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2NhYTY1MDY4ODcyMTQ2NjA4MjhiMDNkYzY4NjM5MTUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzFhYWFmZmU2ZmY0YjQ2ZjZhZDg0MDVkMTlkYTJjOGViID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzIzMjFjNmU4NDRkZDRlNGJiMjQ5MGUzNWIyMGU1Yzk4ID0gJCgnPGRpdiBpZD0iaHRtbF8yMzIxYzZlODQ0ZGQ0ZTRiYjI0OTBlMzViMjBlNWM5OCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TGl0dGxlIFBvcnR1Z2FsLCBUcmluaXR5LCBXZXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzFhYWFmZmU2ZmY0YjQ2ZjZhZDg0MDVkMTlkYTJjOGViLnNldENvbnRlbnQoaHRtbF8yMzIxYzZlODQ0ZGQ0ZTRiYjI0OTBlMzViMjBlNWM5OCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85YjEyNTMwOTE0NGY0MjZlYWMyMmFmNzExNTAyYmVkYi5iaW5kUG9wdXAocG9wdXBfMWFhYWZmZTZmZjRiNDZmNmFkODQwNWQxOWRhMmM4ZWIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMDhhYzY4NmNhZWY2NDE5ZTk3YjQwY2NiMmVjMmUxZjEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42MzY4NDcyLC03OS40MjgxOTE0MDAwMDAwMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jYWE2NTA2ODg3MjE0NjYwODI4YjAzZGM2ODYzOTE1MSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zM2UwZWVhZDc0OGE0ZjhmODhkODBjMjExYzcyOGQzYSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82NTM0MzU5MWQ5YzU0YmNmOTdjYmFmNTdjYjk5ODU2MyA9ICQoJzxkaXYgaWQ9Imh0bWxfNjUzNDM1OTFkOWM1NGJjZjk3Y2JhZjU3Y2I5OTg1NjMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJyb2NrdG9uLCBFeGhpYml0aW9uIFBsYWNlLCBQYXJrZGFsZSBWaWxsYWdlLCBXZXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzMzZTBlZWFkNzQ4YTRmOGY4OGQ4MGMyMTFjNzI4ZDNhLnNldENvbnRlbnQoaHRtbF82NTM0MzU5MWQ5YzU0YmNmOTdjYmFmNTdjYjk5ODU2Myk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wOGFjNjg2Y2FlZjY0MTllOTdiNDBjY2IyZWMyZTFmMS5iaW5kUG9wdXAocG9wdXBfMzNlMGVlYWQ3NDhhNGY4Zjg4ZDgwYzIxMWM3MjhkM2EpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYjcxZjU4NDY3NDUyNDNmYWI5ZDcyZjZkODhlMDQ1MzMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjE2MDgzLC03OS40NjQ3NjMyOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jYWE2NTA2ODg3MjE0NjYwODI4YjAzZGM2ODYzOTE1MSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83YThiN2FkMmRhYjM0NDE1YjI2ZTQzZjM4YzI1N2VmOCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85ZmRmZDM0ODE3Njc0MGQ1YjdiZTg2ZTQwYWQ0NzkzYyA9ICQoJzxkaXYgaWQ9Imh0bWxfOWZkZmQzNDgxNzY3NDBkNWI3YmU4NmU0MGFkNDc5M2MiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkhpZ2ggUGFyaywgVGhlIEp1bmN0aW9uIFNvdXRoLCBXZXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzdhOGI3YWQyZGFiMzQ0MTViMjZlNDNmMzhjMjU3ZWY4LnNldENvbnRlbnQoaHRtbF85ZmRmZDM0ODE3Njc0MGQ1YjdiZTg2ZTQwYWQ0NzkzYyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iNzFmNTg0Njc0NTI0M2ZhYjlkNzJmNmQ4OGUwNDUzMy5iaW5kUG9wdXAocG9wdXBfN2E4YjdhZDJkYWIzNDQxNWIyNmU0M2YzOGMyNTdlZjgpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZTcxNmVlOGMzOGM4NDQ2NDgxYjE4ZDY5NGE0M2RkYTggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDg5NTk3LC03OS40NTYzMjVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfY2FhNjUwNjg4NzIxNDY2MDgyOGIwM2RjNjg2MzkxNTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNWY4NGFiMDdlODgyNGY3MDgwZmEwODY5NWMxOGUwMzggPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNGI2YzBiMDEyNzMwNDlmYzkwMDQxNTVlYmYwNTJlMGMgPSAkKCc8ZGl2IGlkPSJodG1sXzRiNmMwYjAxMjczMDQ5ZmM5MDA0MTU1ZWJmMDUyZTBjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5QYXJrZGFsZSwgUm9uY2VzdmFsbGVzLCBXZXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzVmODRhYjA3ZTg4MjRmNzA4MGZhMDg2OTVjMThlMDM4LnNldENvbnRlbnQoaHRtbF80YjZjMGIwMTI3MzA0OWZjOTAwNDE1NWViZjA1MmUwYyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lNzE2ZWU4YzM4Yzg0NDY0ODFiMThkNjk0YTQzZGRhOC5iaW5kUG9wdXAocG9wdXBfNWY4NGFiMDdlODgyNGY3MDgwZmEwODY5NWMxOGUwMzgpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMDE0YTIyYTBjNjkzNGNmYjg5ZGQ3MjNmYTViYTA0NWYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTE1NzA2LC03OS40ODQ0NDk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2NhYTY1MDY4ODcyMTQ2NjA4MjhiMDNkYzY4NjM5MTUxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Q3ZWYwMzIyMWQwZjRjYTk4YjI5YWNhYTNiYTNkNTI5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2EwNmZlMTVkOTdiMzQ2ZDdhZTIyZTEwYzBiNWJjYTI0ID0gJCgnPGRpdiBpZD0iaHRtbF9hMDZmZTE1ZDk3YjM0NmQ3YWUyMmUxMGMwYjViY2EyNCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UnVubnltZWRlLCBTd2Fuc2VhLCBXZXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Q3ZWYwMzIyMWQwZjRjYTk4YjI5YWNhYTNiYTNkNTI5LnNldENvbnRlbnQoaHRtbF9hMDZmZTE1ZDk3YjM0NmQ3YWUyMmUxMGMwYjViY2EyNCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wMTRhMjJhMGM2OTM0Y2ZiODlkZDcyM2ZhNWJhMDQ1Zi5iaW5kUG9wdXAocG9wdXBfZDdlZjAzMjIxZDBmNGNhOThiMjlhY2FhM2JhM2Q1MjkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMmM5NGFhMWE2ODk2NDQ5YThmMjcwYzVjNjYwZTE1MjkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjI3NDM5LC03OS4zMjE1NThdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfY2FhNjUwNjg4NzIxNDY2MDgyOGIwM2RjNjg2MzkxNTEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYjY1ZWI5YTNlOWY2NDgxOTkzYzllNmU3YjJlYTBmYzMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZDE0MDkwMTRiNmNhNDZjYTkxN2ExYWU2NGM0ZmY3NGYgPSAkKCc8ZGl2IGlkPSJodG1sX2QxNDA5MDE0YjZjYTQ2Y2E5MTdhMWFlNjRjNGZmNzRmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5CdXNpbmVzcyBSZXBseSBNYWlsIFByb2Nlc3NpbmcgQ2VudHJlIDk2OSBFYXN0ZXJuLCBFYXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2I2NWViOWEzZTlmNjQ4MTk5M2M5ZTZlN2IyZWEwZmMzLnNldENvbnRlbnQoaHRtbF9kMTQwOTAxNGI2Y2E0NmNhOTE3YTFhZTY0YzRmZjc0Zik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yYzk0YWExYTY4OTY0NDlhOGYyNzBjNWM2NjBlMTUyOS5iaW5kUG9wdXAocG9wdXBfYjY1ZWI5YTNlOWY2NDgxOTkzYzllNmU3YjJlYTBmYzMpOwoKICAgICAgICAgICAgCiAgICAgICAgCjwvc2NyaXB0Pg==" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



### begin to explore, get nearby venue information of each neighborhood


```python
CLIENT_ID = '3ZXMWIQUJV4T0PEWKY1JJRAVVBPA1NY5OL3G3HBNJGRY4HWT' # your Foursquare ID
CLIENT_SECRET = 'VF5OJMMW03GHLSIDNPWBYUPNCNEKB55GRLIYUASE0D5LZPUP' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version
```


```python
LIMIT=100

def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)
```


```python
toronto_venues = getNearbyVenues(names=neighborhoods['Neighborhood'],
                                   latitudes=neighborhoods['Latitude'],
                                   longitudes=neighborhoods['Longitude']
                                  )
```

    The Beaches
    The Danforth West, Riverdale
    The Beaches West, India Bazaar
    Studio District
    Lawrence Park
    Davisville North
    North Toronto West
    Davisville
    Moore Park, Summerhill East
    Deer Park, Forest Hill SE, Rathnelly, South Hill, Summerhill West
    Rosedale
    Cabbagetown, St. James Town
    Church and Wellesley
    Harbourfront, Regent Park
    Ryerson, Garden District
    St. James Town
    Berczy Park
    Central Bay Street
    Adelaide, King, Richmond
    Harbourfront East, Toronto Islands, Union Station
    Design Exchange, Toronto Dominion Centre
    Commerce Court, Victoria Hotel
    Roselawn
    Forest Hill North, Forest Hill West
    The Annex, North Midtown, Yorkville
    Harbord, University of Toronto
    Chinatown, Grange Park, Kensington Market
    CN Tower, Bathurst Quay, Island airport, Harbourfront West, King and Spadina, Railway Lands, South Niagara
    Stn A PO Boxes 25 The Esplanade
    First Canadian Place, Underground city
    Christie
    Dovercourt Village, Dufferin
    Little Portugal, Trinity
    Brockton, Exhibition Place, Parkdale Village
    High Park, The Junction South
    Parkdale, Roncesvalles
    Runnymede, Swansea
    Business Reply Mail Processing Centre 969 Eastern
    

### find 1707 venues in total


```python
print(toronto_venues.shape)
toronto_venues
```

    (1707, 7)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>Neighborhood Latitude</th>
      <th>Neighborhood Longitude</th>
      <th>Venue</th>
      <th>Venue Latitude</th>
      <th>Venue Longitude</th>
      <th>Venue Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Beaches</td>
      <td>43.676357</td>
      <td>-79.293031</td>
      <td>Grover Pub and Grub</td>
      <td>43.679181</td>
      <td>-79.297215</td>
      <td>Pub</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Beaches</td>
      <td>43.676357</td>
      <td>-79.293031</td>
      <td>Starbucks</td>
      <td>43.678798</td>
      <td>-79.298045</td>
      <td>Coffee Shop</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Beaches</td>
      <td>43.676357</td>
      <td>-79.293031</td>
      <td>Upper Beaches</td>
      <td>43.680563</td>
      <td>-79.292869</td>
      <td>Neighborhood</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The Beaches</td>
      <td>43.676357</td>
      <td>-79.293031</td>
      <td>Beaches Fitness</td>
      <td>43.680319</td>
      <td>-79.290991</td>
      <td>Gym / Fitness Center</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The Beaches</td>
      <td>43.676357</td>
      <td>-79.293031</td>
      <td>Dip 'n Sip</td>
      <td>43.678897</td>
      <td>-79.297745</td>
      <td>Coffee Shop</td>
    </tr>
    <tr>
      <th>5</th>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>Pantheon</td>
      <td>43.677621</td>
      <td>-79.351434</td>
      <td>Greek Restaurant</td>
    </tr>
    <tr>
      <th>6</th>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>Dolce Gelato</td>
      <td>43.677773</td>
      <td>-79.351187</td>
      <td>Ice Cream Shop</td>
    </tr>
    <tr>
      <th>7</th>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>MenEssentials</td>
      <td>43.677820</td>
      <td>-79.351265</td>
      <td>Cosmetics Shop</td>
    </tr>
    <tr>
      <th>8</th>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>Messini Authentic Gyros</td>
      <td>43.677827</td>
      <td>-79.350569</td>
      <td>Greek Restaurant</td>
    </tr>
    <tr>
      <th>9</th>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>Mezes</td>
      <td>43.677962</td>
      <td>-79.350196</td>
      <td>Greek Restaurant</td>
    </tr>
    <tr>
      <th>10</th>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>Cafe Fiorentina</td>
      <td>43.677743</td>
      <td>-79.350115</td>
      <td>Italian Restaurant</td>
    </tr>
    <tr>
      <th>11</th>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>Christina's On The Danforth</td>
      <td>43.678240</td>
      <td>-79.349185</td>
      <td>Greek Restaurant</td>
    </tr>
    <tr>
      <th>12</th>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>Moksha Yoga Danforth</td>
      <td>43.677622</td>
      <td>-79.352116</td>
      <td>Yoga Studio</td>
    </tr>
    <tr>
      <th>13</th>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>The Big Carrot Natural Food Market</td>
      <td>43.677631</td>
      <td>-79.353076</td>
      <td>Health Food Store</td>
    </tr>
    <tr>
      <th>14</th>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>Louis Cifer Brew Works</td>
      <td>43.677663</td>
      <td>-79.351313</td>
      <td>Brewery</td>
    </tr>
    <tr>
      <th>15</th>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>The Auld Spot Pub</td>
      <td>43.677335</td>
      <td>-79.353130</td>
      <td>Pub</td>
    </tr>
    <tr>
      <th>16</th>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>7 Numbers</td>
      <td>43.677062</td>
      <td>-79.353934</td>
      <td>Italian Restaurant</td>
    </tr>
    <tr>
      <th>17</th>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>Valley Farm Produce</td>
      <td>43.677999</td>
      <td>-79.349969</td>
      <td>Fruit &amp; Vegetable Store</td>
    </tr>
    <tr>
      <th>18</th>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>La Diperie</td>
      <td>43.677530</td>
      <td>-79.352295</td>
      <td>Ice Cream Shop</td>
    </tr>
    <tr>
      <th>19</th>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>Pizzeria Libretto</td>
      <td>43.678489</td>
      <td>-79.347576</td>
      <td>Pizza Place</td>
    </tr>
    <tr>
      <th>20</th>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>Rikkochez</td>
      <td>43.677267</td>
      <td>-79.353274</td>
      <td>Restaurant</td>
    </tr>
    <tr>
      <th>21</th>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>Pan on the Danforth</td>
      <td>43.678263</td>
      <td>-79.348648</td>
      <td>Greek Restaurant</td>
    </tr>
    <tr>
      <th>22</th>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>Astoria Shish Kebob House</td>
      <td>43.677689</td>
      <td>-79.351892</td>
      <td>Greek Restaurant</td>
    </tr>
    <tr>
      <th>23</th>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>The Big Carrot Organic Juice Bar</td>
      <td>43.677502</td>
      <td>-79.352776</td>
      <td>Juice Bar</td>
    </tr>
    <tr>
      <th>24</th>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>Don Valley Trail</td>
      <td>43.676331</td>
      <td>-79.353923</td>
      <td>Trail</td>
    </tr>
    <tr>
      <th>25</th>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>Re: Reading</td>
      <td>43.678507</td>
      <td>-79.347678</td>
      <td>Bookstore</td>
    </tr>
    <tr>
      <th>26</th>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>Ouzeri</td>
      <td>43.678193</td>
      <td>-79.348908</td>
      <td>Greek Restaurant</td>
    </tr>
    <tr>
      <th>27</th>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>Detroit Eatery</td>
      <td>43.677594</td>
      <td>-79.351761</td>
      <td>Diner</td>
    </tr>
    <tr>
      <th>28</th>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>Caffé Demetre</td>
      <td>43.677683</td>
      <td>-79.351608</td>
      <td>Dessert Shop</td>
    </tr>
    <tr>
      <th>29</th>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>Tsaa Tea Shop</td>
      <td>43.677769</td>
      <td>-79.351304</td>
      <td>Bubble Tea Shop</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1677</th>
      <td>Runnymede, Swansea</td>
      <td>43.651571</td>
      <td>-79.484450</td>
      <td>Pizza Pizza</td>
      <td>43.649311</td>
      <td>-79.484560</td>
      <td>Pizza Place</td>
    </tr>
    <tr>
      <th>1678</th>
      <td>Runnymede, Swansea</td>
      <td>43.651571</td>
      <td>-79.484450</td>
      <td>A Dark Horse</td>
      <td>43.649533</td>
      <td>-79.483056</td>
      <td>Bar</td>
    </tr>
    <tr>
      <th>1679</th>
      <td>Runnymede, Swansea</td>
      <td>43.651571</td>
      <td>-79.484450</td>
      <td>Second Cup</td>
      <td>43.650137</td>
      <td>-79.480795</td>
      <td>Café</td>
    </tr>
    <tr>
      <th>1680</th>
      <td>Runnymede, Swansea</td>
      <td>43.651571</td>
      <td>-79.484450</td>
      <td>The Station</td>
      <td>43.649456</td>
      <td>-79.484667</td>
      <td>Diner</td>
    </tr>
    <tr>
      <th>1681</th>
      <td>Runnymede, Swansea</td>
      <td>43.651571</td>
      <td>-79.484450</td>
      <td>Strictly Bulk</td>
      <td>43.649937</td>
      <td>-79.482366</td>
      <td>Food &amp; Drink Shop</td>
    </tr>
    <tr>
      <th>1682</th>
      <td>Runnymede, Swansea</td>
      <td>43.651571</td>
      <td>-79.484450</td>
      <td>Pizzaiolo</td>
      <td>43.649395</td>
      <td>-79.483560</td>
      <td>Pizza Place</td>
    </tr>
    <tr>
      <th>1683</th>
      <td>Runnymede, Swansea</td>
      <td>43.651571</td>
      <td>-79.484450</td>
      <td>Classico</td>
      <td>43.649194</td>
      <td>-79.484699</td>
      <td>Pizza Place</td>
    </tr>
    <tr>
      <th>1684</th>
      <td>Runnymede, Swansea</td>
      <td>43.651571</td>
      <td>-79.484450</td>
      <td>The Body Shop</td>
      <td>43.650173</td>
      <td>-79.481635</td>
      <td>Cosmetics Shop</td>
    </tr>
    <tr>
      <th>1685</th>
      <td>Runnymede, Swansea</td>
      <td>43.651571</td>
      <td>-79.484450</td>
      <td>Supper Solved</td>
      <td>43.648781</td>
      <td>-79.485233</td>
      <td>Restaurant</td>
    </tr>
    <tr>
      <th>1686</th>
      <td>Runnymede, Swansea</td>
      <td>43.651571</td>
      <td>-79.484450</td>
      <td>Kingsway Meat Products &amp; Deli</td>
      <td>43.650299</td>
      <td>-79.480827</td>
      <td>Butcher</td>
    </tr>
    <tr>
      <th>1687</th>
      <td>Runnymede, Swansea</td>
      <td>43.651571</td>
      <td>-79.484450</td>
      <td>Think Fitness</td>
      <td>43.647966</td>
      <td>-79.486462</td>
      <td>Gym</td>
    </tr>
    <tr>
      <th>1688</th>
      <td>Runnymede, Swansea</td>
      <td>43.651571</td>
      <td>-79.484450</td>
      <td>Supplements Plus</td>
      <td>43.650512</td>
      <td>-79.479262</td>
      <td>Supplement Shop</td>
    </tr>
    <tr>
      <th>1689</th>
      <td>Runnymede, Swansea</td>
      <td>43.651571</td>
      <td>-79.484450</td>
      <td>Los Arrieros</td>
      <td>43.655593</td>
      <td>-79.487028</td>
      <td>South American Restaurant</td>
    </tr>
    <tr>
      <th>1690</th>
      <td>Business Reply Mail Processing Centre 969 Eastern</td>
      <td>43.662744</td>
      <td>-79.321558</td>
      <td>Rorschach Brewing Co.</td>
      <td>43.663483</td>
      <td>-79.319824</td>
      <td>Brewery</td>
    </tr>
    <tr>
      <th>1691</th>
      <td>Business Reply Mail Processing Centre 969 Eastern</td>
      <td>43.662744</td>
      <td>-79.321558</td>
      <td>Leslieville Farmers Market</td>
      <td>43.664901</td>
      <td>-79.319784</td>
      <td>Farmers Market</td>
    </tr>
    <tr>
      <th>1692</th>
      <td>Business Reply Mail Processing Centre 969 Eastern</td>
      <td>43.662744</td>
      <td>-79.321558</td>
      <td>Queen Margherita Pizza</td>
      <td>43.664685</td>
      <td>-79.324164</td>
      <td>Pizza Place</td>
    </tr>
    <tr>
      <th>1693</th>
      <td>Business Reply Mail Processing Centre 969 Eastern</td>
      <td>43.662744</td>
      <td>-79.321558</td>
      <td>The Sidekick</td>
      <td>43.664484</td>
      <td>-79.325162</td>
      <td>Comic Shop</td>
    </tr>
    <tr>
      <th>1694</th>
      <td>Business Reply Mail Processing Centre 969 Eastern</td>
      <td>43.662744</td>
      <td>-79.321558</td>
      <td>Chino Locos</td>
      <td>43.664653</td>
      <td>-79.325584</td>
      <td>Burrito Place</td>
    </tr>
    <tr>
      <th>1695</th>
      <td>Business Reply Mail Processing Centre 969 Eastern</td>
      <td>43.662744</td>
      <td>-79.321558</td>
      <td>The Green Wood</td>
      <td>43.664728</td>
      <td>-79.324117</td>
      <td>Restaurant</td>
    </tr>
    <tr>
      <th>1696</th>
      <td>Business Reply Mail Processing Centre 969 Eastern</td>
      <td>43.662744</td>
      <td>-79.321558</td>
      <td>Chick-n-Joy</td>
      <td>43.665181</td>
      <td>-79.321403</td>
      <td>Fast Food Restaurant</td>
    </tr>
    <tr>
      <th>1697</th>
      <td>Business Reply Mail Processing Centre 969 Eastern</td>
      <td>43.662744</td>
      <td>-79.321558</td>
      <td>Ashbridges Bay Skatepark</td>
      <td>43.662548</td>
      <td>-79.315631</td>
      <td>Skate Park</td>
    </tr>
    <tr>
      <th>1698</th>
      <td>Business Reply Mail Processing Centre 969 Eastern</td>
      <td>43.662744</td>
      <td>-79.321558</td>
      <td>East End Garden Centre &amp; Hardware</td>
      <td>43.664555</td>
      <td>-79.324598</td>
      <td>Garden Center</td>
    </tr>
    <tr>
      <th>1699</th>
      <td>Business Reply Mail Processing Centre 969 Eastern</td>
      <td>43.662744</td>
      <td>-79.321558</td>
      <td>Amin Car Repair Garage</td>
      <td>43.663544</td>
      <td>-79.320130</td>
      <td>Auto Workshop</td>
    </tr>
    <tr>
      <th>1700</th>
      <td>Business Reply Mail Processing Centre 969 Eastern</td>
      <td>43.662744</td>
      <td>-79.321558</td>
      <td>The Ashbridge Estate</td>
      <td>43.664691</td>
      <td>-79.321805</td>
      <td>Garden</td>
    </tr>
    <tr>
      <th>1701</th>
      <td>Business Reply Mail Processing Centre 969 Eastern</td>
      <td>43.662744</td>
      <td>-79.321558</td>
      <td>TTC Russell Division</td>
      <td>43.664908</td>
      <td>-79.322560</td>
      <td>Light Rail Station</td>
    </tr>
    <tr>
      <th>1702</th>
      <td>Business Reply Mail Processing Centre 969 Eastern</td>
      <td>43.662744</td>
      <td>-79.321558</td>
      <td>Jonathan Ashbridge Park</td>
      <td>43.664702</td>
      <td>-79.319898</td>
      <td>Park</td>
    </tr>
    <tr>
      <th>1703</th>
      <td>Business Reply Mail Processing Centre 969 Eastern</td>
      <td>43.662744</td>
      <td>-79.321558</td>
      <td>The Ten Spot</td>
      <td>43.664815</td>
      <td>-79.324213</td>
      <td>Spa</td>
    </tr>
    <tr>
      <th>1704</th>
      <td>Business Reply Mail Processing Centre 969 Eastern</td>
      <td>43.662744</td>
      <td>-79.321558</td>
      <td>Toronto Yoga Mamas</td>
      <td>43.664824</td>
      <td>-79.324335</td>
      <td>Yoga Studio</td>
    </tr>
    <tr>
      <th>1705</th>
      <td>Business Reply Mail Processing Centre 969 Eastern</td>
      <td>43.662744</td>
      <td>-79.321558</td>
      <td>Greenwood Cigar &amp; Variety</td>
      <td>43.664538</td>
      <td>-79.325379</td>
      <td>Smoke Shop</td>
    </tr>
    <tr>
      <th>1706</th>
      <td>Business Reply Mail Processing Centre 969 Eastern</td>
      <td>43.662744</td>
      <td>-79.321558</td>
      <td>Revolution Recording</td>
      <td>43.662561</td>
      <td>-79.326940</td>
      <td>Recording Studio</td>
    </tr>
  </tbody>
</table>
<p>1707 rows × 7 columns</p>
</div>



### obtain venue counts for each neighborhood


```python
toronto_venues.groupby('Neighborhood').count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood Latitude</th>
      <th>Neighborhood Longitude</th>
      <th>Venue</th>
      <th>Venue Latitude</th>
      <th>Venue Longitude</th>
      <th>Venue Category</th>
    </tr>
    <tr>
      <th>Neighborhood</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Adelaide, King, Richmond</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Berczy Park</th>
      <td>54</td>
      <td>54</td>
      <td>54</td>
      <td>54</td>
      <td>54</td>
      <td>54</td>
    </tr>
    <tr>
      <th>Brockton, Exhibition Place, Parkdale Village</th>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
    </tr>
    <tr>
      <th>Business Reply Mail Processing Centre 969 Eastern</th>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
    </tr>
    <tr>
      <th>CN Tower, Bathurst Quay, Island airport, Harbourfront West, King and Spadina, Railway Lands, South Niagara</th>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
    </tr>
    <tr>
      <th>Cabbagetown, St. James Town</th>
      <td>47</td>
      <td>47</td>
      <td>47</td>
      <td>47</td>
      <td>47</td>
      <td>47</td>
    </tr>
    <tr>
      <th>Central Bay Street</th>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
    </tr>
    <tr>
      <th>Chinatown, Grange Park, Kensington Market</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Christie</th>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
    </tr>
    <tr>
      <th>Church and Wellesley</th>
      <td>87</td>
      <td>87</td>
      <td>87</td>
      <td>87</td>
      <td>87</td>
      <td>87</td>
    </tr>
    <tr>
      <th>Commerce Court, Victoria Hotel</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Davisville</th>
      <td>36</td>
      <td>36</td>
      <td>36</td>
      <td>36</td>
      <td>36</td>
      <td>36</td>
    </tr>
    <tr>
      <th>Davisville North</th>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>Deer Park, Forest Hill SE, Rathnelly, South Hill, Summerhill West</th>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
    </tr>
    <tr>
      <th>Design Exchange, Toronto Dominion Centre</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Dovercourt Village, Dufferin</th>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
    </tr>
    <tr>
      <th>First Canadian Place, Underground city</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Forest Hill North, Forest Hill West</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Harbord, University of Toronto</th>
      <td>34</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
    </tr>
    <tr>
      <th>Harbourfront East, Toronto Islands, Union Station</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Harbourfront, Regent Park</th>
      <td>49</td>
      <td>49</td>
      <td>49</td>
      <td>49</td>
      <td>49</td>
      <td>49</td>
    </tr>
    <tr>
      <th>High Park, The Junction South</th>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
    </tr>
    <tr>
      <th>Lawrence Park</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Little Portugal, Trinity</th>
      <td>66</td>
      <td>66</td>
      <td>66</td>
      <td>66</td>
      <td>66</td>
      <td>66</td>
    </tr>
    <tr>
      <th>Moore Park, Summerhill East</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>North Toronto West</th>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
    </tr>
    <tr>
      <th>Parkdale, Roncesvalles</th>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
    </tr>
    <tr>
      <th>Rosedale</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Roselawn</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Runnymede, Swansea</th>
      <td>40</td>
      <td>40</td>
      <td>40</td>
      <td>40</td>
      <td>40</td>
      <td>40</td>
    </tr>
    <tr>
      <th>Ryerson, Garden District</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>St. James Town</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Stn A PO Boxes 25 The Esplanade</th>
      <td>95</td>
      <td>95</td>
      <td>95</td>
      <td>95</td>
      <td>95</td>
      <td>95</td>
    </tr>
    <tr>
      <th>Studio District</th>
      <td>39</td>
      <td>39</td>
      <td>39</td>
      <td>39</td>
      <td>39</td>
      <td>39</td>
    </tr>
    <tr>
      <th>The Annex, North Midtown, Yorkville</th>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
    </tr>
    <tr>
      <th>The Beaches</th>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>The Beaches West, India Bazaar</th>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
    </tr>
    <tr>
      <th>The Danforth West, Riverdale</th>
      <td>42</td>
      <td>42</td>
      <td>42</td>
      <td>42</td>
      <td>42</td>
      <td>42</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('There are {} uniques categories.'.format(len(toronto_venues['Venue Category'].unique())))
```

    There are 238 uniques categories.
    


```python
# one hot encoding
toronto_onehot = pd.get_dummies(toronto_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
toronto_onehot['Neighborhood'] = toronto_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [toronto_onehot.columns[-1]] + list(toronto_onehot.columns[:-1])
toronto_onehot = toronto_onehot[fixed_columns]

toronto_onehot.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Yoga Studio</th>
      <th>Adult Boutique</th>
      <th>Afghan Restaurant</th>
      <th>Airport</th>
      <th>Airport Food Court</th>
      <th>Airport Gate</th>
      <th>Airport Lounge</th>
      <th>Airport Service</th>
      <th>Airport Terminal</th>
      <th>American Restaurant</th>
      <th>...</th>
      <th>Thrift / Vintage Store</th>
      <th>Toy / Game Store</th>
      <th>Trail</th>
      <th>Train Station</th>
      <th>Vegetarian / Vegan Restaurant</th>
      <th>Video Game Store</th>
      <th>Vietnamese Restaurant</th>
      <th>Wine Bar</th>
      <th>Wings Joint</th>
      <th>Women's Store</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 238 columns</p>
</div>



### all 1707 venues were coded one-hot


```python
toronto_onehot.shape
```




    (1707, 238)




```python
toronto_grouped = toronto_onehot.groupby('Neighborhood').mean().reset_index()
toronto_grouped
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>Yoga Studio</th>
      <th>Adult Boutique</th>
      <th>Afghan Restaurant</th>
      <th>Airport</th>
      <th>Airport Food Court</th>
      <th>Airport Gate</th>
      <th>Airport Lounge</th>
      <th>Airport Service</th>
      <th>Airport Terminal</th>
      <th>...</th>
      <th>Thrift / Vintage Store</th>
      <th>Toy / Game Store</th>
      <th>Trail</th>
      <th>Train Station</th>
      <th>Vegetarian / Vegan Restaurant</th>
      <th>Video Game Store</th>
      <th>Vietnamese Restaurant</th>
      <th>Wine Bar</th>
      <th>Wings Joint</th>
      <th>Women's Store</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adelaide, King, Richmond</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Berczy Park</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brockton, Exhibition Place, Parkdale Village</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Business Reply Mail Processing Centre 969 Eastern</td>
      <td>0.058824</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CN Tower, Bathurst Quay, Island airport, Harbo...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.142857</td>
      <td>0.142857</td>
      <td>0.142857</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Cabbagetown, St. James Town</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Central Bay Street</td>
      <td>0.012195</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.012195</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012195</td>
      <td>0.000000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Chinatown, Grange Park, Kensington Market</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Christie</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Church and Wellesley</td>
      <td>0.011494</td>
      <td>0.011494</td>
      <td>0.011494</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.011494</td>
      <td>0.011494</td>
      <td>0.000000</td>
      <td>0.011494</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Commerce Court, Victoria Hotel</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Davisville</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.027778</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Davisville North</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Deer Park, Forest Hill SE, Rathnelly, South Hi...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.071429</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Design Exchange, Toronto Dominion Centre</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Dovercourt Village, Dufferin</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>16</th>
      <td>First Canadian Place, Underground city</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.01</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Forest Hill North, Forest Hill West</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.25000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Harbord, University of Toronto</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.029412</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Harbourfront East, Toronto Islands, Union Station</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.01</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Harbourfront, Regent Park</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>21</th>
      <td>High Park, The Junction South</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Lawrence Park</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Little Portugal, Trinity</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.030303</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Moore Park, Summerhill East</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>25</th>
      <td>North Toronto West</td>
      <td>0.047619</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Parkdale, Roncesvalles</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Rosedale</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.25000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Roselawn</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Runnymede, Swansea</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.025000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Ryerson, Garden District</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>31</th>
      <td>St. James Town</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Stn A PO Boxes 25 The Esplanade</td>
      <td>0.010526</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Studio District</td>
      <td>0.025641</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>34</th>
      <td>The Annex, North Midtown, Yorkville</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>35</th>
      <td>The Beaches</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>36</th>
      <td>The Beaches West, India Bazaar</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>37</th>
      <td>The Danforth West, Riverdale</td>
      <td>0.023810</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.02381</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
<p>38 rows × 238 columns</p>
</div>




```python
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]
```

### find top 10 venues of each neighborhood


```python
num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = toronto_grouped['Neighborhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adelaide, King, Richmond</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Thai Restaurant</td>
      <td>Steakhouse</td>
      <td>American Restaurant</td>
      <td>Hotel</td>
      <td>Restaurant</td>
      <td>Asian Restaurant</td>
      <td>Bar</td>
      <td>Gym</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Berczy Park</td>
      <td>Coffee Shop</td>
      <td>Restaurant</td>
      <td>Cocktail Bar</td>
      <td>Cheese Shop</td>
      <td>Bakery</td>
      <td>Italian Restaurant</td>
      <td>Beer Bar</td>
      <td>Steakhouse</td>
      <td>Seafood Restaurant</td>
      <td>Pub</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brockton, Exhibition Place, Parkdale Village</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Breakfast Spot</td>
      <td>Bar</td>
      <td>Stadium</td>
      <td>Italian Restaurant</td>
      <td>Burrito Place</td>
      <td>Climbing Gym</td>
      <td>Office</td>
      <td>Furniture / Home Store</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Business Reply Mail Processing Centre 969 Eastern</td>
      <td>Yoga Studio</td>
      <td>Auto Workshop</td>
      <td>Park</td>
      <td>Pizza Place</td>
      <td>Recording Studio</td>
      <td>Restaurant</td>
      <td>Burrito Place</td>
      <td>Brewery</td>
      <td>Skate Park</td>
      <td>Smoke Shop</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CN Tower, Bathurst Quay, Island airport, Harbo...</td>
      <td>Airport Lounge</td>
      <td>Airport Terminal</td>
      <td>Airport Service</td>
      <td>Boutique</td>
      <td>Sculpture Garden</td>
      <td>Plane</td>
      <td>Boat or Ferry</td>
      <td>Airport Gate</td>
      <td>Airport Food Court</td>
      <td>Airport</td>
    </tr>
  </tbody>
</table>
</div>



### begin clustering, set clusters to 3


```python
# set number of clusters
kclusters = 3

toronto_grouped_clustering = toronto_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=np.random.randint(1000)).fit(toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 
```




    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])




```python
# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

toronto_merged = neighborhoods

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
toronto_merged = toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

toronto_merged.head() # check the last columns!
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M4E</td>
      <td>East Toronto</td>
      <td>The Beaches</td>
      <td>43.676357</td>
      <td>-79.293031</td>
      <td>1</td>
      <td>Coffee Shop</td>
      <td>Gym / Fitness Center</td>
      <td>Pub</td>
      <td>Women's Store</td>
      <td>Discount Store</td>
      <td>Fast Food Restaurant</td>
      <td>Farmers Market</td>
      <td>Falafel Restaurant</td>
      <td>Event Space</td>
      <td>Ethiopian Restaurant</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M4K</td>
      <td>East Toronto</td>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>1</td>
      <td>Greek Restaurant</td>
      <td>Coffee Shop</td>
      <td>Ice Cream Shop</td>
      <td>Bookstore</td>
      <td>Italian Restaurant</td>
      <td>Yoga Studio</td>
      <td>Cosmetics Shop</td>
      <td>Brewery</td>
      <td>Bubble Tea Shop</td>
      <td>Restaurant</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M4L</td>
      <td>East Toronto</td>
      <td>The Beaches West, India Bazaar</td>
      <td>43.668999</td>
      <td>-79.315572</td>
      <td>1</td>
      <td>Park</td>
      <td>Sushi Restaurant</td>
      <td>Board Shop</td>
      <td>Brewery</td>
      <td>Burger Joint</td>
      <td>Sandwich Place</td>
      <td>Burrito Place</td>
      <td>Pub</td>
      <td>Pet Store</td>
      <td>Pizza Place</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M4M</td>
      <td>East Toronto</td>
      <td>Studio District</td>
      <td>43.659526</td>
      <td>-79.340923</td>
      <td>1</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>Bakery</td>
      <td>Italian Restaurant</td>
      <td>American Restaurant</td>
      <td>Yoga Studio</td>
      <td>Coworking Space</td>
      <td>Seafood Restaurant</td>
      <td>Sandwich Place</td>
      <td>Cheese Shop</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M4N</td>
      <td>Central Toronto</td>
      <td>Lawrence Park</td>
      <td>43.728020</td>
      <td>-79.388790</td>
      <td>1</td>
      <td>Bus Line</td>
      <td>Park</td>
      <td>Swim School</td>
      <td>Dim Sum Restaurant</td>
      <td>Women's Store</td>
      <td>Fast Food Restaurant</td>
      <td>Farmers Market</td>
      <td>Falafel Restaurant</td>
      <td>Event Space</td>
      <td>Ethiopian Restaurant</td>
    </tr>
  </tbody>
</table>
</div>



### visualize clustering results


```python
# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=12)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['Neighborhood'], toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><iframe src="data:text/html;charset=utf-8;base64,PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVMgPSBmYWxzZTsgTF9OT19UT1VDSCA9IGZhbHNlOyBMX0RJU0FCTEVfM0QgPSBmYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2FqYXguZ29vZ2xlYXBpcy5jb20vYWpheC9saWJzL2pxdWVyeS8xLjExLjEvanF1ZXJ5Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9yYXdnaXQuY29tL3B5dGhvbi12aXN1YWxpemF0aW9uL2ZvbGl1bS9tYXN0ZXIvZm9saXVtL3RlbXBsYXRlcy9sZWFmbGV0LmF3ZXNvbWUucm90YXRlLmNzcyIvPgogICAgPHN0eWxlPmh0bWwsIGJvZHkge3dpZHRoOiAxMDAlO2hlaWdodDogMTAwJTttYXJnaW46IDA7cGFkZGluZzogMDt9PC9zdHlsZT4KICAgIDxzdHlsZT4jbWFwIHtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MDtib3R0b206MDtyaWdodDowO2xlZnQ6MDt9PC9zdHlsZT4KICAgIAogICAgICAgICAgICA8c3R5bGU+ICNtYXBfNjBhM2RmYWZhNWRkNDczOWJhODdjNjZjZDI1NjIxYzEgewogICAgICAgICAgICAgICAgcG9zaXRpb24gOiByZWxhdGl2ZTsKICAgICAgICAgICAgICAgIHdpZHRoIDogMTAwLjAlOwogICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgICAgICAgICAgdG9wOiAwLjAlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICA8L3N0eWxlPgogICAgICAgIAo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgICAgICAgICA8ZGl2IGNsYXNzPSJmb2xpdW0tbWFwIiBpZD0ibWFwXzYwYTNkZmFmYTVkZDQ3MzliYTg3YzY2Y2QyNTYyMWMxIiA+PC9kaXY+CiAgICAgICAgCjwvYm9keT4KPHNjcmlwdD4gICAgCiAgICAKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGJvdW5kcyA9IG51bGw7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIG1hcF82MGEzZGZhZmE1ZGQ0NzM5YmE4N2M2NmNkMjU2MjFjMSA9IEwubWFwKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ21hcF82MGEzZGZhZmE1ZGQ0NzM5YmE4N2M2NmNkMjU2MjFjMScsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7Y2VudGVyOiBbNDMuNjgsLTc5LjM5XSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHpvb206IDEyLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbWF4Qm91bmRzOiBib3VuZHMsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBsYXllcnM6IFtdLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgd29ybGRDb3B5SnVtcDogZmFsc2UsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBjcnM6IEwuQ1JTLkVQU0czODU3CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIH0pOwogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgdGlsZV9sYXllcl82Zjk5MTNjNjhiYTk0NzRkOGE2MWFiYTRjNmNlOTAyZiA9IEwudGlsZUxheWVyKAogICAgICAgICAgICAgICAgJ2h0dHBzOi8ve3N9LnRpbGUub3BlbnN0cmVldG1hcC5vcmcve3p9L3t4fS97eX0ucG5nJywKICAgICAgICAgICAgICAgIHsKICAiYXR0cmlidXRpb24iOiBudWxsLAogICJkZXRlY3RSZXRpbmEiOiBmYWxzZSwKICAibWF4Wm9vbSI6IDE4LAogICJtaW5ab29tIjogMSwKICAibm9XcmFwIjogZmFsc2UsCiAgInN1YmRvbWFpbnMiOiAiYWJjIgp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82MGEzZGZhZmE1ZGQ0NzM5YmE4N2M2NmNkMjU2MjFjMSk7CiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMmYxZWEwY2Y0NGExNGY4ZDk3NWUxNjZjZWM1Mzg3OWYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NzYzNTczOTk5OTk5OSwtNzkuMjkzMDMxMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82MGEzZGZhZmE1ZGQ0NzM5YmE4N2M2NmNkMjU2MjFjMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xMzJlYjI4ZGVkNjA0N2Y3YTIwNTE0OTQ1N2I5ZDYwMiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85NzMyNTA2YmQwOGM0ZWM2YjRiMWY0NDFhOTRiOThkZiA9ICQoJzxkaXYgaWQ9Imh0bWxfOTczMjUwNmJkMDhjNGVjNmI0YjFmNDQxYTk0Yjk4ZGYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRoZSBCZWFjaGVzIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMTMyZWIyOGRlZDYwNDdmN2EyMDUxNDk0NTdiOWQ2MDIuc2V0Q29udGVudChodG1sXzk3MzI1MDZiZDA4YzRlYzZiNGIxZjQ0MWE5NGI5OGRmKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzJmMWVhMGNmNDRhMTRmOGQ5NzVlMTY2Y2VjNTM4NzlmLmJpbmRQb3B1cChwb3B1cF8xMzJlYjI4ZGVkNjA0N2Y3YTIwNTE0OTQ1N2I5ZDYwMik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82MDgwOWY1YWEzMTA0NDdmODE2Yjk2NWUyNDQ3ZDA0MSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY3OTU1NzEsLTc5LjM1MjE4OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82MGEzZGZhZmE1ZGQ0NzM5YmE4N2M2NmNkMjU2MjFjMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zNjFlYTI5NmRlZjA0MGFkODg2NDFlNzNjMGY5NjkyNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iNTg1NzA4ZWFhNmE0MjJlYjgyNjk4MGI5MjY4ODU2YiA9ICQoJzxkaXYgaWQ9Imh0bWxfYjU4NTcwOGVhYTZhNDIyZWI4MjY5ODBiOTI2ODg1NmIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRoZSBEYW5mb3J0aCBXZXN0LCBSaXZlcmRhbGUgQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zNjFlYTI5NmRlZjA0MGFkODg2NDFlNzNjMGY5NjkyNi5zZXRDb250ZW50KGh0bWxfYjU4NTcwOGVhYTZhNDIyZWI4MjY5ODBiOTI2ODg1NmIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNjA4MDlmNWFhMzEwNDQ3ZjgxNmI5NjVlMjQ0N2QwNDEuYmluZFBvcHVwKHBvcHVwXzM2MWVhMjk2ZGVmMDQwYWQ4ODY0MWU3M2MwZjk2OTI2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2ZjNWUzNjZjNjRhNjRkNTI4NzY1Zjc4YzU3YzJhNWViID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjY4OTk4NSwtNzkuMzE1NTcxNTk5OTk5OThdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjBhM2RmYWZhNWRkNDczOWJhODdjNjZjZDI1NjIxYzEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNzUzNjE2MDRmNWViNDI3ZGFiNGVlMzFlMGMwYzc5YzUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfY2FiNDVkOWE5YmQ5NDIzYmIwMzhjMjA0ODM3ZmJlZjcgPSAkKCc8ZGl2IGlkPSJodG1sX2NhYjQ1ZDlhOWJkOTQyM2JiMDM4YzIwNDgzN2ZiZWY3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UaGUgQmVhY2hlcyBXZXN0LCBJbmRpYSBCYXphYXIgQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83NTM2MTYwNGY1ZWI0MjdkYWI0ZWUzMWUwYzBjNzljNS5zZXRDb250ZW50KGh0bWxfY2FiNDVkOWE5YmQ5NDIzYmIwMzhjMjA0ODM3ZmJlZjcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZmM1ZTM2NmM2NGE2NGQ1Mjg3NjVmNzhjNTdjMmE1ZWIuYmluZFBvcHVwKHBvcHVwXzc1MzYxNjA0ZjVlYjQyN2RhYjRlZTMxZTBjMGM3OWM1KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzZmMjdmMTZiMDc1MjRlZWY4YzgxMTAwNTY5NzQ3YzM1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU5NTI1NSwtNzkuMzQwOTIzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzYwYTNkZmFmYTVkZDQ3MzliYTg3YzY2Y2QyNTYyMWMxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2FlNGQ0ZTA5YTgwYjQ5ODdiN2RiZDQ4ZjAzMjY3NjNiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2ZiZDIwYzQ4MDQxOTRhYjliOTE1NmZiZjkzNTQyZTYwID0gJCgnPGRpdiBpZD0iaHRtbF9mYmQyMGM0ODA0MTk0YWI5YjkxNTZmYmY5MzU0MmU2MCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U3R1ZGlvIERpc3RyaWN0IENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYWU0ZDRlMDlhODBiNDk4N2I3ZGJkNDhmMDMyNjc2M2Iuc2V0Q29udGVudChodG1sX2ZiZDIwYzQ4MDQxOTRhYjliOTE1NmZiZjkzNTQyZTYwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzZmMjdmMTZiMDc1MjRlZWY4YzgxMTAwNTY5NzQ3YzM1LmJpbmRQb3B1cChwb3B1cF9hZTRkNGUwOWE4MGI0OTg3YjdkYmQ0OGYwMzI2NzYzYik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80ZDU0NjlmZWZiNzg0ODExOWQ3ZjVhNGQ1N2U5OGY1OSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcyODAyMDUsLTc5LjM4ODc5MDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjBhM2RmYWZhNWRkNDczOWJhODdjNjZjZDI1NjIxYzEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMTEzNDE4ZjcyODUwNGM0MDgwZTUyZjdmNjcxZmVkNDMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNzQzMzg3NDk5NDI4NDk0Yzg3MGE3ODAzYzMzNTgzZmQgPSAkKCc8ZGl2IGlkPSJodG1sXzc0MzM4NzQ5OTQyODQ5NGM4NzBhNzgwM2MzMzU4M2ZkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5MYXdyZW5jZSBQYXJrIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMTEzNDE4ZjcyODUwNGM0MDgwZTUyZjdmNjcxZmVkNDMuc2V0Q29udGVudChodG1sXzc0MzM4NzQ5OTQyODQ5NGM4NzBhNzgwM2MzMzU4M2ZkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzRkNTQ2OWZlZmI3ODQ4MTE5ZDdmNWE0ZDU3ZTk4ZjU5LmJpbmRQb3B1cChwb3B1cF8xMTM0MThmNzI4NTA0YzQwODBlNTJmN2Y2NzFmZWQ0Myk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mZjUxZWU0ODAwNzI0YjU2OGJmNDExMmUyN2VlZDNhYiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcxMjc1MTEsLTc5LjM5MDE5NzVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjBhM2RmYWZhNWRkNDczOWJhODdjNjZjZDI1NjIxYzEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNmQ2YTgxMjBhYTM4NDgwMmE2YjUwNWI3YjMxNTQyNDAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZjk2Nzk2NGY4ZDQ0NDE1Y2I2YzJjODE5YjAyOTdlNjIgPSAkKCc8ZGl2IGlkPSJodG1sX2Y5Njc5NjRmOGQ0NDQxNWNiNmMyYzgxOWIwMjk3ZTYyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5EYXZpc3ZpbGxlIE5vcnRoIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNmQ2YTgxMjBhYTM4NDgwMmE2YjUwNWI3YjMxNTQyNDAuc2V0Q29udGVudChodG1sX2Y5Njc5NjRmOGQ0NDQxNWNiNmMyYzgxOWIwMjk3ZTYyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2ZmNTFlZTQ4MDA3MjRiNTY4YmY0MTEyZTI3ZWVkM2FiLmJpbmRQb3B1cChwb3B1cF82ZDZhODEyMGFhMzg0ODAyYTZiNTA1YjdiMzE1NDI0MCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wNWM3MDZjMTk5OTg0OTdhYmQyOGFiNWRmZDgyNmM4NSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcxNTM4MzQsLTc5LjQwNTY3ODQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzYwYTNkZmFmYTVkZDQ3MzliYTg3YzY2Y2QyNTYyMWMxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzRhNTBlYThhMDgyNzQ0YTE4NTE0OTE2MGFmYTdlNmI2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzI5M2FkYmE4ZWNlNzQ3ZDliNTc0NzBlMzg1NGI2ODk4ID0gJCgnPGRpdiBpZD0iaHRtbF8yOTNhZGJhOGVjZTc0N2Q5YjU3NDcwZTM4NTRiNjg5OCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Tm9ydGggVG9yb250byBXZXN0IENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNGE1MGVhOGEwODI3NDRhMTg1MTQ5MTYwYWZhN2U2YjYuc2V0Q29udGVudChodG1sXzI5M2FkYmE4ZWNlNzQ3ZDliNTc0NzBlMzg1NGI2ODk4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzA1YzcwNmMxOTk5ODQ5N2FiZDI4YWI1ZGZkODI2Yzg1LmJpbmRQb3B1cChwb3B1cF80YTUwZWE4YTA4Mjc0NGExODUxNDkxNjBhZmE3ZTZiNik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hZmZmYTM1ZTVlODI0NTRhOGIwZDNiNmYzMGFlNDU3OSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcwNDMyNDQsLTc5LjM4ODc5MDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjBhM2RmYWZhNWRkNDczOWJhODdjNjZjZDI1NjIxYzEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZDFlYmJhZWVhZjE4NGFiZjhmM2RhNzRjNjlmNDc0ZWQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMjRiYzc1NDU1ZGJhNDNkMGI0MGM5MDViYzc2MjUyMjkgPSAkKCc8ZGl2IGlkPSJodG1sXzI0YmM3NTQ1NWRiYTQzZDBiNDBjOTA1YmM3NjI1MjI5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5EYXZpc3ZpbGxlIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZDFlYmJhZWVhZjE4NGFiZjhmM2RhNzRjNjlmNDc0ZWQuc2V0Q29udGVudChodG1sXzI0YmM3NTQ1NWRiYTQzZDBiNDBjOTA1YmM3NjI1MjI5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2FmZmZhMzVlNWU4MjQ1NGE4YjBkM2I2ZjMwYWU0NTc5LmJpbmRQb3B1cChwb3B1cF9kMWViYmFlZWFmMTg0YWJmOGYzZGE3NGM2OWY0NzRlZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mYjg5ZDA2ZDMzMWM0MzhmOTI4OWViOWZmYzlhZTVmNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY4OTU3NDMsLTc5LjM4MzE1OTkwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzYwYTNkZmFmYTVkZDQ3MzliYTg3YzY2Y2QyNTYyMWMxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzhkZmJlYjIxZDczYjQ5ZGJhYTM5MDg4ZTBiZGZiNzE3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzdlNGNlZjRjZWMwODRmNmU4OGEyNTI3MzNiMjIyYTQ5ID0gJCgnPGRpdiBpZD0iaHRtbF83ZTRjZWY0Y2VjMDg0ZjZlODhhMjUyNzMzYjIyMmE0OSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TW9vcmUgUGFyaywgU3VtbWVyaGlsbCBFYXN0IENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOGRmYmViMjFkNzNiNDlkYmFhMzkwODhlMGJkZmI3MTcuc2V0Q29udGVudChodG1sXzdlNGNlZjRjZWMwODRmNmU4OGEyNTI3MzNiMjIyYTQ5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2ZiODlkMDZkMzMxYzQzOGY5Mjg5ZWI5ZmZjOWFlNWY2LmJpbmRQb3B1cChwb3B1cF84ZGZiZWIyMWQ3M2I0OWRiYWEzOTA4OGUwYmRmYjcxNyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mZTQ0MmFiMWMzMWU0MDA2YjgyMmZkYmM4NzFjMzJmZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY4NjQxMjI5OTk5OTk5LC03OS40MDAwNDkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzYwYTNkZmFmYTVkZDQ3MzliYTg3YzY2Y2QyNTYyMWMxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2E4NmM0OWM3OGI3ZjRmNTA4MDg3YzcxOGY0OTQ0MWEwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2QyZWIxYTViZWFmZjQ5ZDRiNjFmZTE3ZDE1OWFlMDE3ID0gJCgnPGRpdiBpZD0iaHRtbF9kMmViMWE1YmVhZmY0OWQ0YjYxZmUxN2QxNTlhZTAxNyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RGVlciBQYXJrLCBGb3Jlc3QgSGlsbCBTRSwgUmF0aG5lbGx5LCBTb3V0aCBIaWxsLCBTdW1tZXJoaWxsIFdlc3QgQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hODZjNDljNzhiN2Y0ZjUwODA4N2M3MThmNDk0NDFhMC5zZXRDb250ZW50KGh0bWxfZDJlYjFhNWJlYWZmNDlkNGI2MWZlMTdkMTU5YWUwMTcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZmU0NDJhYjFjMzFlNDAwNmI4MjJmZGJjODcxYzMyZmUuYmluZFBvcHVwKHBvcHVwX2E4NmM0OWM3OGI3ZjRmNTA4MDg3YzcxOGY0OTQ0MWEwKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2YxMjYxMDMyYTM0NjQ2OGU5ZDZjOGFjMzQyNGJmZDBiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjc5NTYyNiwtNzkuMzc3NTI5NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjBhM2RmYWZhNWRkNDczOWJhODdjNjZjZDI1NjIxYzEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfY2VlZWIxNjc4MjcyNGUxY2EyN2E2NTVhZTZlNGNiMzYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMDMxZTQwYzk1YzRhNGU0NzhkZjFhMjdhZTFkMGY1MDUgPSAkKCc8ZGl2IGlkPSJodG1sXzAzMWU0MGM5NWM0YTRlNDc4ZGYxYTI3YWUxZDBmNTA1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Sb3NlZGFsZSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2NlZWViMTY3ODI3MjRlMWNhMjdhNjU1YWU2ZTRjYjM2LnNldENvbnRlbnQoaHRtbF8wMzFlNDBjOTVjNGE0ZTQ3OGRmMWEyN2FlMWQwZjUwNSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mMTI2MTAzMmEzNDY0NjhlOWQ2YzhhYzM0MjRiZmQwYi5iaW5kUG9wdXAocG9wdXBfY2VlZWIxNjc4MjcyNGUxY2EyN2E2NTVhZTZlNGNiMzYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzRjNmYxMGNkMjBhNDliYjgzZTdkYzExMjRkMmMxYjcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Njc5NjcsLTc5LjM2NzY3NTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjBhM2RmYWZhNWRkNDczOWJhODdjNjZjZDI1NjIxYzEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZDYyMWY5ZTczNDM5NDEyODkyYThjMjFlYjBjNTM0MzMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMmRhNjY4YTdmOGQ1NGQxZWJlNTE1YmVhZjQwOGNiMjMgPSAkKCc8ZGl2IGlkPSJodG1sXzJkYTY2OGE3ZjhkNTRkMWViZTUxNWJlYWY0MDhjYjIzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DYWJiYWdldG93biwgU3QuIEphbWVzIFRvd24gQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kNjIxZjllNzM0Mzk0MTI4OTJhOGMyMWViMGM1MzQzMy5zZXRDb250ZW50KGh0bWxfMmRhNjY4YTdmOGQ1NGQxZWJlNTE1YmVhZjQwOGNiMjMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYzRjNmYxMGNkMjBhNDliYjgzZTdkYzExMjRkMmMxYjcuYmluZFBvcHVwKHBvcHVwX2Q2MjFmOWU3MzQzOTQxMjg5MmE4YzIxZWIwYzUzNDMzKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzlhMjgzN2YxY2Y1NDRhMzNhYWI3NzQxYmQ1Zjk3NjBhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjY1ODU5OSwtNzkuMzgzMTU5OTAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjBhM2RmYWZhNWRkNDczOWJhODdjNjZjZDI1NjIxYzEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZmU0ZTljMzg1ZDRkNGFmZGJhNDc5NjhhYjhhM2MxNDAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZmZkMzIyNTFjYjk2NGMwNjhhZDY1OWE2NjMzYmU5YzMgPSAkKCc8ZGl2IGlkPSJodG1sX2ZmZDMyMjUxY2I5NjRjMDY4YWQ2NTlhNjYzM2JlOWMzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DaHVyY2ggYW5kIFdlbGxlc2xleSBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2ZlNGU5YzM4NWQ0ZDRhZmRiYTQ3OTY4YWI4YTNjMTQwLnNldENvbnRlbnQoaHRtbF9mZmQzMjI1MWNiOTY0YzA2OGFkNjU5YTY2MzNiZTljMyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85YTI4MzdmMWNmNTQ0YTMzYWFiNzc0MWJkNWY5NzYwYS5iaW5kUG9wdXAocG9wdXBfZmU0ZTljMzg1ZDRkNGFmZGJhNDc5NjhhYjhhM2MxNDApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZDVjODUzNzFkNWZmNDM0ZWE5ZDlmNWY0MWRmMTMyYjcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTQyNTk5LC03OS4zNjA2MzU5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzYwYTNkZmFmYTVkZDQ3MzliYTg3YzY2Y2QyNTYyMWMxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzk2MDRlNGM1MDE2ZjRjOGY5N2YxYjFhZTc5MzI4YmQ3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzRkOWUxY2VmMWFhOTQxZjQ4MjkxMjQ0Y2EwZGQ4YjgxID0gJCgnPGRpdiBpZD0iaHRtbF80ZDllMWNlZjFhYTk0MWY0ODI5MTI0NGNhMGRkOGI4MSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SGFyYm91cmZyb250LCBSZWdlbnQgUGFyayBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzk2MDRlNGM1MDE2ZjRjOGY5N2YxYjFhZTc5MzI4YmQ3LnNldENvbnRlbnQoaHRtbF80ZDllMWNlZjFhYTk0MWY0ODI5MTI0NGNhMGRkOGI4MSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kNWM4NTM3MWQ1ZmY0MzRlYTlkOWY1ZjQxZGYxMzJiNy5iaW5kUG9wdXAocG9wdXBfOTYwNGU0YzUwMTZmNGM4Zjk3ZjFiMWFlNzkzMjhiZDcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNGM5MzVhMzQwMDE3NDVkM2FmNTllMmY1MjE4MmMyY2MgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTcxNjE4LC03OS4zNzg5MzcwOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82MGEzZGZhZmE1ZGQ0NzM5YmE4N2M2NmNkMjU2MjFjMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81OGU3YWIyZjMyNWY0ZDllOGFiNWU5NGFjNmVlZjZkZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xNDMzNjJkY2E3NjI0NzM0YjZlN2FhOTdkYzUyZGQ3MyA9ICQoJzxkaXYgaWQ9Imh0bWxfMTQzMzYyZGNhNzYyNDczNGI2ZTdhYTk3ZGM1MmRkNzMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJ5ZXJzb24sIEdhcmRlbiBEaXN0cmljdCBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzU4ZTdhYjJmMzI1ZjRkOWU4YWI1ZTk0YWM2ZWVmNmRmLnNldENvbnRlbnQoaHRtbF8xNDMzNjJkY2E3NjI0NzM0YjZlN2FhOTdkYzUyZGQ3Myk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80YzkzNWEzNDAwMTc0NWQzYWY1OWUyZjUyMTgyYzJjYy5iaW5kUG9wdXAocG9wdXBfNThlN2FiMmYzMjVmNGQ5ZThhYjVlOTRhYzZlZWY2ZGYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNDRhMDNjMjA4ZjE3NGUzYmI5YTk2NmU2ZjBmODAwYmMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTE0OTM5LC03OS4zNzU0MTc5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzYwYTNkZmFmYTVkZDQ3MzliYTg3YzY2Y2QyNTYyMWMxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzZhODVmNWViZjljNjQzZGY5NTBjNTE4OGU2M2NlNDVjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzk1ZmU0OWFhZTFmYjQ4NmM4NzBlMmZmN2YxMTM1OTczID0gJCgnPGRpdiBpZD0iaHRtbF85NWZlNDlhYWUxZmI0ODZjODcwZTJmZjdmMTEzNTk3MyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U3QuIEphbWVzIFRvd24gQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82YTg1ZjVlYmY5YzY0M2RmOTUwYzUxODhlNjNjZTQ1Yy5zZXRDb250ZW50KGh0bWxfOTVmZTQ5YWFlMWZiNDg2Yzg3MGUyZmY3ZjExMzU5NzMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNDRhMDNjMjA4ZjE3NGUzYmI5YTk2NmU2ZjBmODAwYmMuYmluZFBvcHVwKHBvcHVwXzZhODVmNWViZjljNjQzZGY5NTBjNTE4OGU2M2NlNDVjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2MyZTNlZDE1MTA2NzQ5YWNiODI5NzA2YjUyYTYyNjUwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ0NzcwNzk5OTk5OTk2LC03OS4zNzMzMDY0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzYwYTNkZmFmYTVkZDQ3MzliYTg3YzY2Y2QyNTYyMWMxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2UwYTYzMzNiMzVlODQ5NTk5NTE1ODcwOTg0ZTRlZTk0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2NhMjY0MjAyOWMwYTRjZjc4OTgyM2NkODk1YmJiNTBlID0gJCgnPGRpdiBpZD0iaHRtbF9jYTI2NDIwMjljMGE0Y2Y3ODk4MjNjZDg5NWJiYjUwZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QmVyY3p5IFBhcmsgQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lMGE2MzMzYjM1ZTg0OTU5OTUxNTg3MDk4NGU0ZWU5NC5zZXRDb250ZW50KGh0bWxfY2EyNjQyMDI5YzBhNGNmNzg5ODIzY2Q4OTViYmI1MGUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYzJlM2VkMTUxMDY3NDlhY2I4Mjk3MDZiNTJhNjI2NTAuYmluZFBvcHVwKHBvcHVwX2UwYTYzMzNiMzVlODQ5NTk5NTE1ODcwOTg0ZTRlZTk0KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzg4MjFkOTk1OTRjMjQ5MDM5MTk1Mzc5NGI2MzAyYjU4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU3OTUyNCwtNzkuMzg3MzgyNl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82MGEzZGZhZmE1ZGQ0NzM5YmE4N2M2NmNkMjU2MjFjMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81OTVjOGU5NmFlMDI0NzExYjVjOGE3OTQzYmRlMTM4MiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hM2Y1YjJhM2UwNDI0ZWQxOWI0NTliNTcxYWE1YWFhMSA9ICQoJzxkaXYgaWQ9Imh0bWxfYTNmNWIyYTNlMDQyNGVkMTliNDU5YjU3MWFhNWFhYTEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNlbnRyYWwgQmF5IFN0cmVldCBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzU5NWM4ZTk2YWUwMjQ3MTFiNWM4YTc5NDNiZGUxMzgyLnNldENvbnRlbnQoaHRtbF9hM2Y1YjJhM2UwNDI0ZWQxOWI0NTliNTcxYWE1YWFhMSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84ODIxZDk5NTk0YzI0OTAzOTE5NTM3OTRiNjMwMmI1OC5iaW5kUG9wdXAocG9wdXBfNTk1YzhlOTZhZTAyNDcxMWI1YzhhNzk0M2JkZTEzODIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYmI3YWQwYWI0YTU2NGQ4YzkwOGQ3YTgwM2UyZmRjNmYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTA1NzEyMDAwMDAwMSwtNzkuMzg0NTY3NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82MGEzZGZhZmE1ZGQ0NzM5YmE4N2M2NmNkMjU2MjFjMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85NDQwYzMwMjY3ZDY0MTY4YTU3ZmFmNTU1NWZlNDU3YSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84MjI4M2RiYThjNmM0ZGNlOTlhZGFmYTdlZmI1Y2UyOCA9ICQoJzxkaXYgaWQ9Imh0bWxfODIyODNkYmE4YzZjNGRjZTk5YWRhZmE3ZWZiNWNlMjgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkFkZWxhaWRlLCBLaW5nLCBSaWNobW9uZCBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzk0NDBjMzAyNjdkNjQxNjhhNTdmYWY1NTU1ZmU0NTdhLnNldENvbnRlbnQoaHRtbF84MjI4M2RiYThjNmM0ZGNlOTlhZGFmYTdlZmI1Y2UyOCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iYjdhZDBhYjRhNTY0ZDhjOTA4ZDdhODAzZTJmZGM2Zi5iaW5kUG9wdXAocG9wdXBfOTQ0MGMzMDI2N2Q2NDE2OGE1N2ZhZjU1NTVmZTQ1N2EpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTY4NTNmYzgzN2IyNDQzOGEyMDRhOTgwNjRhOWJmMjUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDA4MTU3LC03OS4zODE3NTIyOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82MGEzZGZhZmE1ZGQ0NzM5YmE4N2M2NmNkMjU2MjFjMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81ZGE0MTEwZmIzYjY0NmVkYjllOWE2N2RkYWM3M2UyYiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80NmZlYjdkNjQ3ZTU0MTRiOTZmYmQ5ZWVhZWJjMDg2YyA9ICQoJzxkaXYgaWQ9Imh0bWxfNDZmZWI3ZDY0N2U1NDE0Yjk2ZmJkOWVlYWViYzA4NmMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkhhcmJvdXJmcm9udCBFYXN0LCBUb3JvbnRvIElzbGFuZHMsIFVuaW9uIFN0YXRpb24gQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81ZGE0MTEwZmIzYjY0NmVkYjllOWE2N2RkYWM3M2UyYi5zZXRDb250ZW50KGh0bWxfNDZmZWI3ZDY0N2U1NDE0Yjk2ZmJkOWVlYWViYzA4NmMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNTY4NTNmYzgzN2IyNDQzOGEyMDRhOTgwNjRhOWJmMjUuYmluZFBvcHVwKHBvcHVwXzVkYTQxMTBmYjNiNjQ2ZWRiOWU5YTY3ZGRhYzczZTJiKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzkwZWVkYTI4M2JiMjQyZTM4MjBiNzgwZWI2Y2Q1NGYyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ3MTc2OCwtNzkuMzgxNTc2NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjBhM2RmYWZhNWRkNDczOWJhODdjNjZjZDI1NjIxYzEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMDBmOGU5Y2VjOTE5NDJjMDgzZDA0ODkyNzlhYTIxNTggPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNjFkMjE3ZTJiYmZkNDViYTljNTg0ZjU5ZWEwN2E0ZDQgPSAkKCc8ZGl2IGlkPSJodG1sXzYxZDIxN2UyYmJmZDQ1YmE5YzU4NGY1OWVhMDdhNGQ0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5EZXNpZ24gRXhjaGFuZ2UsIFRvcm9udG8gRG9taW5pb24gQ2VudHJlIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMDBmOGU5Y2VjOTE5NDJjMDgzZDA0ODkyNzlhYTIxNTguc2V0Q29udGVudChodG1sXzYxZDIxN2UyYmJmZDQ1YmE5YzU4NGY1OWVhMDdhNGQ0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzkwZWVkYTI4M2JiMjQyZTM4MjBiNzgwZWI2Y2Q1NGYyLmJpbmRQb3B1cChwb3B1cF8wMGY4ZTljZWM5MTk0MmMwODNkMDQ4OTI3OWFhMjE1OCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mYTk2NGFkZjAwNjA0ZGViOWRiMGU1MDI5OGNjMTA3ZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0ODE5ODUsLTc5LjM3OTgxNjkwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzYwYTNkZmFmYTVkZDQ3MzliYTg3YzY2Y2QyNTYyMWMxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2IxM2U3MmEyZjNlMDRlNzY5OWZhODg1NDViZDU2OTRhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzFjYmU3ZmQwNTViZDQwMjA5NzE2ZDE4MTMwY2M3NGQ2ID0gJCgnPGRpdiBpZD0iaHRtbF8xY2JlN2ZkMDU1YmQ0MDIwOTcxNmQxODEzMGNjNzRkNiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q29tbWVyY2UgQ291cnQsIFZpY3RvcmlhIEhvdGVsIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYjEzZTcyYTJmM2UwNGU3Njk5ZmE4ODU0NWJkNTY5NGEuc2V0Q29udGVudChodG1sXzFjYmU3ZmQwNTViZDQwMjA5NzE2ZDE4MTMwY2M3NGQ2KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2ZhOTY0YWRmMDA2MDRkZWI5ZGIwZTUwMjk4Y2MxMDdmLmJpbmRQb3B1cChwb3B1cF9iMTNlNzJhMmYzZTA0ZTc2OTlmYTg4NTQ1YmQ1Njk0YSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84YmNlNzFlZGU5ZDY0MTRmYjU3NDQ5NjA2ZDQyOGFjNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcxMTY5NDgsLTc5LjQxNjkzNTU5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MGZmYjQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODBmZmI0IiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzYwYTNkZmFmYTVkZDQ3MzliYTg3YzY2Y2QyNTYyMWMxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2EyOWE0OTcwYTA5MDQ1NGQ5YWE0MDI2NWRmZDkwZjY3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzgxYjY1YjhiMTRhMTQ1ZWU5YzQxZTczZTBhMjQzYTc1ID0gJCgnPGRpdiBpZD0iaHRtbF84MWI2NWI4YjE0YTE0NWVlOWM0MWU3M2UwYTI0M2E3NSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Um9zZWxhd24gQ2x1c3RlciAyPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hMjlhNDk3MGEwOTA0NTRkOWFhNDAyNjVkZmQ5MGY2Ny5zZXRDb250ZW50KGh0bWxfODFiNjViOGIxNGExNDVlZTljNDFlNzNlMGEyNDNhNzUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOGJjZTcxZWRlOWQ2NDE0ZmI1NzQ0OTYwNmQ0MjhhYzYuYmluZFBvcHVwKHBvcHVwX2EyOWE0OTcwYTA5MDQ1NGQ5YWE0MDI2NWRmZDkwZjY3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2Y5MzJjZjQzZTM5NzQ3YWJiMzQ1ODY5MDVkZDIwYmZhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjk2OTQ3NiwtNzkuNDExMzA3MjAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjBhM2RmYWZhNWRkNDczOWJhODdjNjZjZDI1NjIxYzEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZjczYzQ4ZTk1ZGEzNGUyYWFhZGU1NmUyMDI1YmQwODUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzY2NTkwZjkwOTNhNGU1Y2IyOWE0MWU1NTIwZDMxNTkgPSAkKCc8ZGl2IGlkPSJodG1sXzM2NjU5MGY5MDkzYTRlNWNiMjlhNDFlNTUyMGQzMTU5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Gb3Jlc3QgSGlsbCBOb3J0aCwgRm9yZXN0IEhpbGwgV2VzdCBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Y3M2M0OGU5NWRhMzRlMmFhYWRlNTZlMjAyNWJkMDg1LnNldENvbnRlbnQoaHRtbF8zNjY1OTBmOTA5M2E0ZTVjYjI5YTQxZTU1MjBkMzE1OSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mOTMyY2Y0M2UzOTc0N2FiYjM0NTg2OTA1ZGQyMGJmYS5iaW5kUG9wdXAocG9wdXBfZjczYzQ4ZTk1ZGEzNGUyYWFhZGU1NmUyMDI1YmQwODUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMWRlMTg3ODEwZjA0NGE5NWE2ZTBjNDZjNmQ5ZTY3MGUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NzI3MDk3LC03OS40MDU2Nzg0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82MGEzZGZhZmE1ZGQ0NzM5YmE4N2M2NmNkMjU2MjFjMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hZGM2MjFiNjIwMDA0NDVmYmYyODVjYjdjMjFlMDg0ZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xODc4ZGRjYTdmYmE0ZDI3ODc0ZTJiYmJkYTIzZGQ3YiA9ICQoJzxkaXYgaWQ9Imh0bWxfMTg3OGRkY2E3ZmJhNGQyNzg3NGUyYmJiZGEyM2RkN2IiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRoZSBBbm5leCwgTm9ydGggTWlkdG93biwgWW9ya3ZpbGxlIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYWRjNjIxYjYyMDAwNDQ1ZmJmMjg1Y2I3YzIxZTA4NGYuc2V0Q29udGVudChodG1sXzE4NzhkZGNhN2ZiYTRkMjc4NzRlMmJiYmRhMjNkZDdiKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzFkZTE4NzgxMGYwNDRhOTVhNmUwYzQ2YzZkOWU2NzBlLmJpbmRQb3B1cChwb3B1cF9hZGM2MjFiNjIwMDA0NDVmYmYyODVjYjdjMjFlMDg0Zik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iM2ZiNjBhY2YxNzE0ZDhjOThlOThiOWUxZmJjNWNjYSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2MjY5NTYsLTc5LjQwMDA0OTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjBhM2RmYWZhNWRkNDczOWJhODdjNjZjZDI1NjIxYzEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMDE0ZGY1OTU0ODkyNDIzY2JjMmMwOTZiNGEyYTU1MTYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfODk4MzFmNGUzYWRjNGRkODkzMmQzMDc3YmY0ZDI0ZDggPSAkKCc8ZGl2IGlkPSJodG1sXzg5ODMxZjRlM2FkYzRkZDg5MzJkMzA3N2JmNGQyNGQ4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IYXJib3JkLCBVbml2ZXJzaXR5IG9mIFRvcm9udG8gQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wMTRkZjU5NTQ4OTI0MjNjYmMyYzA5NmI0YTJhNTUxNi5zZXRDb250ZW50KGh0bWxfODk4MzFmNGUzYWRjNGRkODkzMmQzMDc3YmY0ZDI0ZDgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYjNmYjYwYWNmMTcxNGQ4Yzk4ZTk4YjllMWZiYzVjY2EuYmluZFBvcHVwKHBvcHVwXzAxNGRmNTk1NDg5MjQyM2NiYzJjMDk2YjRhMmE1NTE2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQyNTMwODA2NzM3MjQyNThhZWMyYTk1MzhkMWZmNGRhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUzMjA1NywtNzkuNDAwMDQ5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82MGEzZGZhZmE1ZGQ0NzM5YmE4N2M2NmNkMjU2MjFjMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xNTVjMGQ4MDdkNzU0ZDkxOTFmOTMzY2JkM2E0ZTY1OSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wODc3YjI2ZTNjMGY0ZWQwYTExOWUzMDM5ZDRiNzQzOCA9ICQoJzxkaXYgaWQ9Imh0bWxfMDg3N2IyNmUzYzBmNGVkMGExMTllMzAzOWQ0Yjc0MzgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNoaW5hdG93biwgR3JhbmdlIFBhcmssIEtlbnNpbmd0b24gTWFya2V0IENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMTU1YzBkODA3ZDc1NGQ5MTkxZjkzM2NiZDNhNGU2NTkuc2V0Q29udGVudChodG1sXzA4NzdiMjZlM2MwZjRlZDBhMTE5ZTMwMzlkNGI3NDM4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzQyNTMwODA2NzM3MjQyNThhZWMyYTk1MzhkMWZmNGRhLmJpbmRQb3B1cChwb3B1cF8xNTVjMGQ4MDdkNzU0ZDkxOTFmOTMzY2JkM2E0ZTY1OSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kN2NiMWNiNmJiODI0ZDVhYmNmNmJjMDIxMGJhY2RkYSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYyODk0NjcsLTc5LjM5NDQxOTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjBhM2RmYWZhNWRkNDczOWJhODdjNjZjZDI1NjIxYzEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfODE2NTIzZTIyMGJkNDE5NmFhMjMzZjI1OTY4N2Q1MTEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOWY5MDRlZTI1N2JkNGI0NGIxN2I2ZjFmZmI0NTgxMzQgPSAkKCc8ZGl2IGlkPSJodG1sXzlmOTA0ZWUyNTdiZDRiNDRiMTdiNmYxZmZiNDU4MTM0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DTiBUb3dlciwgQmF0aHVyc3QgUXVheSwgSXNsYW5kIGFpcnBvcnQsIEhhcmJvdXJmcm9udCBXZXN0LCBLaW5nIGFuZCBTcGFkaW5hLCBSYWlsd2F5IExhbmRzLCBTb3V0aCBOaWFnYXJhIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfODE2NTIzZTIyMGJkNDE5NmFhMjMzZjI1OTY4N2Q1MTEuc2V0Q29udGVudChodG1sXzlmOTA0ZWUyNTdiZDRiNDRiMTdiNmYxZmZiNDU4MTM0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2Q3Y2IxY2I2YmI4MjRkNWFiY2Y2YmMwMjEwYmFjZGRhLmJpbmRQb3B1cChwb3B1cF84MTY1MjNlMjIwYmQ0MTk2YWEyMzNmMjU5Njg3ZDUxMSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zNDAwNTIxYTdlZWU0YTI3OGQ1YmRjMzkwOGM5NjY0MCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NjQzNTIsLTc5LjM3NDg0NTk5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzYwYTNkZmFmYTVkZDQ3MzliYTg3YzY2Y2QyNTYyMWMxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Y4OTUzM2E0YzAyYjQxYjA5ZGM1Yzc3ZGVjZjIwNjRkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzkzMGJmN2YzZDJiYTQ3NDk5MmQzNGU1MGI3YjZmNWNkID0gJCgnPGRpdiBpZD0iaHRtbF85MzBiZjdmM2QyYmE0NzQ5OTJkMzRlNTBiN2I2ZjVjZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U3RuIEEgUE8gQm94ZXMgMjUgVGhlIEVzcGxhbmFkZSBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Y4OTUzM2E0YzAyYjQxYjA5ZGM1Yzc3ZGVjZjIwNjRkLnNldENvbnRlbnQoaHRtbF85MzBiZjdmM2QyYmE0NzQ5OTJkMzRlNTBiN2I2ZjVjZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zNDAwNTIxYTdlZWU0YTI3OGQ1YmRjMzkwOGM5NjY0MC5iaW5kUG9wdXAocG9wdXBfZjg5NTMzYTRjMDJiNDFiMDlkYzVjNzdkZWNmMjA2NGQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNzMxODE0ZGJiNzIzNDQzNDg4MzNlNTg2ZDFjODM0NTcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDg0MjkyLC03OS4zODIyODAyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzYwYTNkZmFmYTVkZDQ3MzliYTg3YzY2Y2QyNTYyMWMxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzI3ODE0M2IwN2UzMTQ0YzU4OTcyYzA1NmZkNDc1MjIxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzhjMzE5NzljNTRlZTQxOGQ5Yzg1YmQ5ZmRhZjY3ODA4ID0gJCgnPGRpdiBpZD0iaHRtbF84YzMxOTc5YzU0ZWU0MThkOWM4NWJkOWZkYWY2NzgwOCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Rmlyc3QgQ2FuYWRpYW4gUGxhY2UsIFVuZGVyZ3JvdW5kIGNpdHkgQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yNzgxNDNiMDdlMzE0NGM1ODk3MmMwNTZmZDQ3NTIyMS5zZXRDb250ZW50KGh0bWxfOGMzMTk3OWM1NGVlNDE4ZDljODViZDlmZGFmNjc4MDgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNzMxODE0ZGJiNzIzNDQzNDg4MzNlNTg2ZDFjODM0NTcuYmluZFBvcHVwKHBvcHVwXzI3ODE0M2IwN2UzMTQ0YzU4OTcyYzA1NmZkNDc1MjIxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2NlMmYyZWU1MWJjYjQzMGI4YmI5NjBhMjQ2YmM5ZjlhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjY5NTQyLC03OS40MjI1NjM3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzYwYTNkZmFmYTVkZDQ3MzliYTg3YzY2Y2QyNTYyMWMxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzVjOWVjNzU4MjA4ODRhMzc4MTI4ZWRhZmVmNzVhODkzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzYyZGY4M2Y5YmZkMzRlMmJhOTI4NzQzNWQ0ZmQ2MDIzID0gJCgnPGRpdiBpZD0iaHRtbF82MmRmODNmOWJmZDM0ZTJiYTkyODc0MzVkNGZkNjAyMyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2hyaXN0aWUgQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81YzllYzc1ODIwODg0YTM3ODEyOGVkYWZlZjc1YTg5My5zZXRDb250ZW50KGh0bWxfNjJkZjgzZjliZmQzNGUyYmE5Mjg3NDM1ZDRmZDYwMjMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfY2UyZjJlZTUxYmNiNDMwYjhiYjk2MGEyNDZiYzlmOWEuYmluZFBvcHVwKHBvcHVwXzVjOWVjNzU4MjA4ODRhMzc4MTI4ZWRhZmVmNzVhODkzKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzIyOGE0NWRjZjM3MDQzZmM4OWJhZDJiYWY2ZTEwYTdhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjY5MDA1MTAwMDAwMDEsLTc5LjQ0MjI1OTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjBhM2RmYWZhNWRkNDczOWJhODdjNjZjZDI1NjIxYzEpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOWZlM2NjMWZlYTRkNGE2MGI0NGU5ZDAxMDYwYWE5MDcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMjBlZmY1YTg5OTg4NGE5MmIzODQ3MDk0Njk0ZWY4MGYgPSAkKCc8ZGl2IGlkPSJodG1sXzIwZWZmNWE4OTk4ODRhOTJiMzg0NzA5NDY5NGVmODBmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3ZlcmNvdXJ0IFZpbGxhZ2UsIER1ZmZlcmluIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOWZlM2NjMWZlYTRkNGE2MGI0NGU5ZDAxMDYwYWE5MDcuc2V0Q29udGVudChodG1sXzIwZWZmNWE4OTk4ODRhOTJiMzg0NzA5NDY5NGVmODBmKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzIyOGE0NWRjZjM3MDQzZmM4OWJhZDJiYWY2ZTEwYTdhLmJpbmRQb3B1cChwb3B1cF85ZmUzY2MxZmVhNGQ0YTYwYjQ0ZTlkMDEwNjBhYTkwNyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kY2E3NDM3MDE1ZWM0Y2I1YjU4MWVjYjExM2VjMmIyMCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NzkyNjcwMDAwMDAwNiwtNzkuNDE5NzQ5N10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82MGEzZGZhZmE1ZGQ0NzM5YmE4N2M2NmNkMjU2MjFjMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zMzVjMWE5OGMzOTY0OTI5YjUwNmNjYTBiYTM0NjQ0NSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hZGVmNGE5YTZkZDg0MWU0OTM5YzUyN2I1MGUyNjZkNCA9ICQoJzxkaXYgaWQ9Imh0bWxfYWRlZjRhOWE2ZGQ4NDFlNDkzOWM1MjdiNTBlMjY2ZDQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkxpdHRsZSBQb3J0dWdhbCwgVHJpbml0eSBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzMzNWMxYTk4YzM5NjQ5MjliNTA2Y2NhMGJhMzQ2NDQ1LnNldENvbnRlbnQoaHRtbF9hZGVmNGE5YTZkZDg0MWU0OTM5YzUyN2I1MGUyNjZkNCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kY2E3NDM3MDE1ZWM0Y2I1YjU4MWVjYjExM2VjMmIyMC5iaW5kUG9wdXAocG9wdXBfMzM1YzFhOThjMzk2NDkyOWI1MDZjY2EwYmEzNDY0NDUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMDY0ZWNiZmVkMjQwNDJjZmI0MmNlZTRlYWUyZmQzOWIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42MzY4NDcyLC03OS40MjgxOTE0MDAwMDAwMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82MGEzZGZhZmE1ZGQ0NzM5YmE4N2M2NmNkMjU2MjFjMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kZjEzYjZhYTMyZmE0NGZhODRmNzU5M2U3YjM4NDExOSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82ZTQ2YzE4Zjc3YzU0YTQzOTQxY2I5OWQzNTFhNGVjNyA9ICQoJzxkaXYgaWQ9Imh0bWxfNmU0NmMxOGY3N2M1NGE0Mzk0MWNiOTlkMzUxYTRlYzciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJyb2NrdG9uLCBFeGhpYml0aW9uIFBsYWNlLCBQYXJrZGFsZSBWaWxsYWdlIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZGYxM2I2YWEzMmZhNDRmYTg0Zjc1OTNlN2IzODQxMTkuc2V0Q29udGVudChodG1sXzZlNDZjMThmNzdjNTRhNDM5NDFjYjk5ZDM1MWE0ZWM3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzA2NGVjYmZlZDI0MDQyY2ZiNDJjZWU0ZWFlMmZkMzliLmJpbmRQb3B1cChwb3B1cF9kZjEzYjZhYTMyZmE0NGZhODRmNzU5M2U3YjM4NDExOSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yYWYyN2M4MTEwNDk0MDdlODlkZDUwOGJjZmJhYzBjNCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2MTYwODMsLTc5LjQ2NDc2MzI5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzYwYTNkZmFmYTVkZDQ3MzliYTg3YzY2Y2QyNTYyMWMxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzYyMWEyYzdiMDk0ZTQ1YzQ5MzBlZGNmNTFmYjRlZDE3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2Q3MmIzMTU4MTRiYTRlNTA5MzdhMzQ3NjU0ODA1NzM1ID0gJCgnPGRpdiBpZD0iaHRtbF9kNzJiMzE1ODE0YmE0ZTUwOTM3YTM0NzY1NDgwNTczNSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SGlnaCBQYXJrLCBUaGUgSnVuY3Rpb24gU291dGggQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82MjFhMmM3YjA5NGU0NWM0OTMwZWRjZjUxZmI0ZWQxNy5zZXRDb250ZW50KGh0bWxfZDcyYjMxNTgxNGJhNGU1MDkzN2EzNDc2NTQ4MDU3MzUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMmFmMjdjODExMDQ5NDA3ZTg5ZGQ1MDhiY2ZiYWMwYzQuYmluZFBvcHVwKHBvcHVwXzYyMWEyYzdiMDk0ZTQ1YzQ5MzBlZGNmNTFmYjRlZDE3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzY0ZWI0MDBlZWVlNjQyYmE4MmVmY2IxNTFiM2RhNzFmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ4OTU5NywtNzkuNDU2MzI1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzYwYTNkZmFmYTVkZDQ3MzliYTg3YzY2Y2QyNTYyMWMxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2QxNmUyYjNhN2YwYjRjNWY5OTVjNWJiYTkxMTFkMzgyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2Q0M2NlMTBlZGVmYjRiZjg4NTA3MDVhZjZmYzQ4ZTcyID0gJCgnPGRpdiBpZD0iaHRtbF9kNDNjZTEwZWRlZmI0YmY4ODUwNzA1YWY2ZmM0OGU3MiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UGFya2RhbGUsIFJvbmNlc3ZhbGxlcyBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2QxNmUyYjNhN2YwYjRjNWY5OTVjNWJiYTkxMTFkMzgyLnNldENvbnRlbnQoaHRtbF9kNDNjZTEwZWRlZmI0YmY4ODUwNzA1YWY2ZmM0OGU3Mik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82NGViNDAwZWVlZTY0MmJhODJlZmNiMTUxYjNkYTcxZi5iaW5kUG9wdXAocG9wdXBfZDE2ZTJiM2E3ZjBiNGM1Zjk5NWM1YmJhOTExMWQzODIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZDIzZDhjMjRkYmI0NDEzMGJkOTUzNzBiM2EzYjljODcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTE1NzA2LC03OS40ODQ0NDk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzYwYTNkZmFmYTVkZDQ3MzliYTg3YzY2Y2QyNTYyMWMxKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2I4NDI0ZTk4NTcxMjQ2ZjBhNzc5MzU4MWM5YzM4NTg2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzY3ZDNkOGQwYjYyOTRiMDY4Y2VmZDA3N2ZhYWRiMGU1ID0gJCgnPGRpdiBpZD0iaHRtbF82N2QzZDhkMGI2Mjk0YjA2OGNlZmQwNzdmYWFkYjBlNSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UnVubnltZWRlLCBTd2Fuc2VhIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYjg0MjRlOTg1NzEyNDZmMGE3NzkzNTgxYzljMzg1ODYuc2V0Q29udGVudChodG1sXzY3ZDNkOGQwYjYyOTRiMDY4Y2VmZDA3N2ZhYWRiMGU1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2QyM2Q4YzI0ZGJiNDQxMzBiZDk1MzcwYjNhM2I5Yzg3LmJpbmRQb3B1cChwb3B1cF9iODQyNGU5ODU3MTI0NmYwYTc3OTM1ODFjOWMzODU4Nik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85YjEzODczYWIxNDE0YmNjOWFlODJkOGI2OTFiZDk2NCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2Mjc0MzksLTc5LjMyMTU1OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82MGEzZGZhZmE1ZGQ0NzM5YmE4N2M2NmNkMjU2MjFjMSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mOGVhMzRjODc4Mzc0ZDljYjQ4ODE5NTdjNmQyMDg5MCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jMmFkY2U5OWY5MjQ0MDViOWFhZDEwZDI0M2M5ZGI3OSA9ICQoJzxkaXYgaWQ9Imh0bWxfYzJhZGNlOTlmOTI0NDA1YjlhYWQxMGQyNDNjOWRiNzkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJ1c2luZXNzIFJlcGx5IE1haWwgUHJvY2Vzc2luZyBDZW50cmUgOTY5IEVhc3Rlcm4gQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mOGVhMzRjODc4Mzc0ZDljYjQ4ODE5NTdjNmQyMDg5MC5zZXRDb250ZW50KGh0bWxfYzJhZGNlOTlmOTI0NDA1YjlhYWQxMGQyNDNjOWRiNzkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOWIxMzg3M2FiMTQxNGJjYzlhZTgyZDhiNjkxYmQ5NjQuYmluZFBvcHVwKHBvcHVwX2Y4ZWEzNGM4NzgzNzRkOWNiNDg4MTk1N2M2ZDIwODkwKTsKCiAgICAgICAgICAgIAogICAgICAgIAo8L3NjcmlwdD4=" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



### inspect the clustering results
#### First cluster has most venues. Those venues look like places people do activities. Typical places include playgrounds, parks, trails, etc.


```python
toronto_merged.loc[toronto_merged['Cluster Labels'] == 0, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]].head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>Central Toronto</td>
      <td>0</td>
      <td>Playground</td>
      <td>Gym</td>
      <td>Tennis Court</td>
      <td>Park</td>
      <td>Farmers Market</td>
      <td>Falafel Restaurant</td>
      <td>Event Space</td>
      <td>Ethiopian Restaurant</td>
      <td>Diner</td>
      <td>Electronics Store</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Downtown Toronto</td>
      <td>0</td>
      <td>Park</td>
      <td>Playground</td>
      <td>Trail</td>
      <td>Diner</td>
      <td>Fast Food Restaurant</td>
      <td>Farmers Market</td>
      <td>Falafel Restaurant</td>
      <td>Event Space</td>
      <td>Ethiopian Restaurant</td>
      <td>Electronics Store</td>
    </tr>
  </tbody>
</table>
</div>



#### Second cluster looks like where people live, since most venues have a combination of resturants, gyms, and certain kinds of shops.


```python
toronto_merged.loc[toronto_merged['Cluster Labels'] == 1, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]].head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>East Toronto</td>
      <td>1</td>
      <td>Coffee Shop</td>
      <td>Gym / Fitness Center</td>
      <td>Pub</td>
      <td>Women's Store</td>
      <td>Discount Store</td>
      <td>Fast Food Restaurant</td>
      <td>Farmers Market</td>
      <td>Falafel Restaurant</td>
      <td>Event Space</td>
      <td>Ethiopian Restaurant</td>
    </tr>
    <tr>
      <th>1</th>
      <td>East Toronto</td>
      <td>1</td>
      <td>Greek Restaurant</td>
      <td>Coffee Shop</td>
      <td>Ice Cream Shop</td>
      <td>Bookstore</td>
      <td>Italian Restaurant</td>
      <td>Yoga Studio</td>
      <td>Cosmetics Shop</td>
      <td>Brewery</td>
      <td>Bubble Tea Shop</td>
      <td>Restaurant</td>
    </tr>
    <tr>
      <th>2</th>
      <td>East Toronto</td>
      <td>1</td>
      <td>Park</td>
      <td>Sushi Restaurant</td>
      <td>Board Shop</td>
      <td>Brewery</td>
      <td>Burger Joint</td>
      <td>Sandwich Place</td>
      <td>Burrito Place</td>
      <td>Pub</td>
      <td>Pet Store</td>
      <td>Pizza Place</td>
    </tr>
    <tr>
      <th>3</th>
      <td>East Toronto</td>
      <td>1</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>Bakery</td>
      <td>Italian Restaurant</td>
      <td>American Restaurant</td>
      <td>Yoga Studio</td>
      <td>Coworking Space</td>
      <td>Seafood Restaurant</td>
      <td>Sandwich Place</td>
      <td>Cheese Shop</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Central Toronto</td>
      <td>1</td>
      <td>Bus Line</td>
      <td>Park</td>
      <td>Swim School</td>
      <td>Dim Sum Restaurant</td>
      <td>Women's Store</td>
      <td>Fast Food Restaurant</td>
      <td>Farmers Market</td>
      <td>Falafel Restaurant</td>
      <td>Event Space</td>
      <td>Ethiopian Restaurant</td>
    </tr>
  </tbody>
</table>
</div>



#### The third cluster looks like a place combining all kinds of activities like shopping, eating, and outdoor stuff.


```python
toronto_merged.loc[toronto_merged['Cluster Labels'] == 2, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]].head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22</th>
      <td>Central Toronto</td>
      <td>2</td>
      <td>Garden</td>
      <td>Pool</td>
      <td>Music Venue</td>
      <td>Women's Store</td>
      <td>Farmers Market</td>
      <td>Falafel Restaurant</td>
      <td>Event Space</td>
      <td>Ethiopian Restaurant</td>
      <td>Electronics Store</td>
      <td>Eastern European Restaurant</td>
    </tr>
  </tbody>
</table>
</div>


