# Load libraries
import requests
import pandas as pd
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import folium
from streamlit_folium import st_folium
import streamlit as st
import ast
import streamlit.components.v1 as components
import time
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from folium import Icon
from optimization import optimize_tsp
from tqdm import tqdm
import itertools
from shapely.geometry import LineString

import pdb

# Function to create a LineString geometry from a string of coordinates
def create_line_string(geometry_str):
    try:
        # Remove any unwanted characters or spaces before parsing
        geometry_str_clean = geometry_str.strip()

        # Check if the string starts and ends with square brackets (list format)
        if geometry_str_clean.startswith('[') and geometry_str_clean.endswith(']'):
            # Safely convert string to list of coordinates
            coords = ast.literal_eval(geometry_str_clean)
            # Create a LineString geometry from the coordinates
            return LineString(coords)
        else:
            raise ValueError(f"Invalid geometry format: {geometry_str}")
    except (ValueError, SyntaxError, TypeError) as e:
        print(f"Error processing geometry: {e}")
        return None  # Return None for invalid geometries

def df_to_gdf(df, route_geometry_col='geometry'):
    df['geometry'] = df[route_geometry_col].apply(lambda x: LineString(x) if isinstance(x, list) else x)
    
    # Crea el GeoDataFrame asegurando que la columna 'geometry' sea la correcta
    gdf_routes = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

    # Set the coordinate reference system (CRS) if known (for example, EPSG:4326 for WGS84)
    gdf_routes.set_crs('EPSG:4326', allow_override=True, inplace=True)

    return gdf_routes


def generate_color_palette(num_colors):
    """Genera una lista de colores de manera gradual"""
    colors = plt.cm.viridis(np.linspace(0, 1, num_colors))
    # Convertir los colores de rgba a hex
    return ['#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255)) for r, g, b, a in colors]

def create_route_map(df_coordinates, df_routes):
    print("Creando el mapa...")
    # Crear el mapa centrado en la primera ruta
    m = folium.Map(tiles='cartodbpositron', location=[df_coordinates['lat'].iloc[0], df_coordinates['lon'].iloc[0]], zoom_start=12.5)

    # Agregar puntos con un ícono de fuego y el texto con el índice
    for i, row in df_coordinates.iterrows():
        popup_text = f'Orden: {i} \\n {row["Falla"]}'

        # Primer punto con un ícono de fuego de color azul
        if i == 0:
            folium.Marker(
                location=[row['lat'], row['lon']],
                icon=folium.Icon(icon="fire", prefix="fa", icon_color='blue'),
                popup=popup_text,
                tooltip=popup_text
            ).add_to(m)
        else:
            folium.Marker(
                location=[row['lat'], row['lon']],
                icon=folium.Icon(icon="fire", prefix="fa", icon_color='red'),
                popup=popup_text,
                tooltip=popup_text
            ).add_to(m)

    # Generar la paleta de colores para las rutas
    num_routes = len(df_routes)
    color_palette = generate_color_palette(num_routes)

    # Iterar sobre el GeoDataFrame y agregar cada ruta como PolyLine
    for idx, row in df_routes.iterrows():
        if row['geometry'].geom_type == 'LineString':
            coordinates = list(row['geometry'].coords)  # Lista de tuplas (lon, lat)

            # Asignar un color de la paleta a esta ruta
            route_color = color_palette[idx]

            # Agregar una PolyLine al mapa con color gradual
            folium.PolyLine(
                locations=[[co[1], co[0]] for co in coordinates],
                color=route_color,
                weight=2.5,
                opacity=1
            ).add_to(m)
    print('mapa creado')
    return m._repr_html_()

# Function to load data
def load_data():
    df_fallas = pd.read_csv("fallas.csv")
    df_routes = pd.read_csv('df_routes.csv')
    df_routes['geometry'] = df_routes['geometry'].apply(ast.literal_eval)
    return df_fallas, df_routes

def add_user_point_to_coordinates(df_fallas, input_lat, input_lon, input_name):
    # Add the user input as the first point
    df_user = pd.DataFrame({
        'Falla': [input_name],
        'lat': [input_lat],
        'lon': [input_lon]
    })
    # Concatenate the user input with the original coordinates
    df_coordinates = pd.concat([df_user, df_fallas], ignore_index=False)
    df_coordinates.reset_index(drop=True, inplace=True)

    return df_coordinates

def create_data_matrix(df_routes):
    index_order = df_routes.ori.unique()

    df_routes = df_routes.pivot(index="ori", columns="des", values="osrm_dist").fillna(0)
    df_routes = df_routes.reindex(index_order, axis=0)  # Reordenar filas
    df_routes = df_routes.reindex(index_order, axis=1)  # Reordenar columnas
    return df_routes

def get_dist_table(df_coordinates, df_routes, id="Falla", route_profile="foot"):
    """
    Obtiene la matriz de distancias y las rutas entre puntos usando OSRM.
    """

    df_dist_query_tbl = df_coordinates.copy()

    # Eliminar duplicados
    df_dist_query_tbl = df_dist_query_tbl.drop_duplicates(subset=["lat", "lon"])

    # Crear todas las combinaciones de pares origen-destino
    df_dist = pd.DataFrame(itertools.product(df_dist_query_tbl[id], repeat=2), columns=["ori", "des"])

    # Agregar columnas vacías para distancia y geometría
    df_dist["osrm_dist"] = None
    df_dist["geometry"] = None  

    table_url = f"http://router.project-osrm.org/table/v1/{route_profile}/"
    route_url = f"http://router.project-osrm.org/route/v1/{route_profile}/"

    while True:
        try:
            #  - Obtener la matriz de distancias
            rutas = [f"{row['lon']},{row['lat']}" for _, row in df_dist_query_tbl.iterrows()]
            rutas = ";".join(rutas)  # Formato para OSRM
            response_table = requests.get(f"{table_url}{rutas}?annotations=distance")
            res_table = response_table.json()

            for io, o in enumerate(df_dist.des.unique()):
                df_dist.loc[df_dist.ori == o, "osrm_dist"] = res_table["distances"][io]

            df_dist = df_dist[df_dist.ori != df_dist.des]  # Eliminar distancias a sí mismo
            # Drop the Fallas existing data

            # Crear un conjunto de pares (ori, des) de df_routes
            existing_pairs = set(zip(df_routes['ori'], df_routes['des']))

            # Filtrar df_dist eliminando aquellos cuyos pares (ori, des) ya estén en df_routes
            df_dist = df_dist[~df_dist.apply(lambda row: (row['ori'], row['des']) in existing_pairs, axis=1)]
           
            # 2️- Obtener rutas individuales para cada par origen-destino
            for index, row in tqdm(df_dist.iterrows()):

                ori_data = df_dist_query_tbl[df_dist_query_tbl[id] == row["ori"]].iloc[0]
                des_data = df_dist_query_tbl[df_dist_query_tbl[id] == row["des"]].iloc[0]

                loc = f"{ori_data['lon']},{ori_data['lat']};{des_data['lon']},{des_data['lat']}"
                response_route = requests.get(f"{route_url}{loc}?alternatives=true&steps=true&geometries=geojson&overview=full")
                res_route = response_route.json()

                # Agregar la geometría de la ruta al DataFrame
                if "routes" in res_route and len(res_route["routes"]) > 0:
                    df_dist.at[index, "geometry"] = res_route["routes"][0]["geometry"]["coordinates"]

            print("Distancias y rutas obtenidas.")
            df_dist = pd.concat([df_dist, df_routes], ignore_index=True)
            df_routes.reset_index(drop=True, inplace=True)
            return df_dist

        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"Error: {e}. \n Retrying in 5 seconds...")
            time.sleep(5)

# Function to create a symmetric distance matrix
def symmetric_min_routes(df_routes):
    df_routes['osrm_dist'] = pd.to_numeric(df_routes['osrm_dist'], errors='coerce')

    for ori in tqdm(df_routes.ori.unique()):
        for des in df_routes.des.unique():
            if ori != des:
                # Filtrar las filas correspondientes al par (ori, des) y (des, ori)
                subset = df_routes[(df_routes.ori.eq(ori) & df_routes.des.eq(des)) | 
                                   (df_routes.ori.eq(des) & df_routes.des.eq(ori))]

                if not subset.empty:
                    # Obtener la distancia mínima y su índice
                    min_idx = subset['osrm_dist'].idxmin()
                    min_dist = subset.at[min_idx, 'osrm_dist']
                    min_geometry = subset.at[min_idx, 'geometry']  # Usa .at en lugar de .values[0]

                    # Actualizar tanto la distancia como la geometría en el DataFrame
                    df_routes.loc[(df_routes.ori.eq(ori) & df_routes.des.eq(des)) | 
                                  (df_routes.ori.eq(des) & df_routes.des.eq(ori)), 
                                  ['osrm_dist']] = min_dist

                    df_routes.loc[(df_routes.ori.eq(ori) & df_routes.des.eq(des)) | 
                                                    (df_routes.ori.eq(des) & df_routes.des.eq(ori)), 
                                                    ['geometry']] = str(min_geometry)
    # Convert to list agai
    df_routes['geometry'] = df_routes['geometry'].apply(ast.literal_eval)
    return df_routes


# Main function
def main():
    # Create the layout
    st.set_page_config( page_icon="🔥", layout="wide")
    st.title("Generador de ruta óptima de Fallas 2025")

    # Inputs del usuario
    input_coor = st.text_input('Introduce las coordenadas de inicio, ej., 39.4674,-0.3771')
    input_name = st.text_input('Introduce el nombre del punto de inicio, e.g., Estación del Nord')

    # Botón para ejecutar la lógica
    execute_button = st.button("Obtener Ruta Óptima")

    # Lógica para generar el mapa solo si se presiona el botón
    if execute_button:
        if input_coor.strip():  # Verificar que el input no esté vacío
            try:
                input_coor = input_coor.replace(' ', '').split(',')
                input_lat = float(input_coor[0])
                input_lon = float(input_coor[1])
                st.write(f"Starting at {input_name}")  # Mostrar el texto

                ###### Optimize the route
                print('loading data...')
                df_fallas, df_routes = load_data()
                df_fallas = add_user_point_to_coordinates(df_fallas, input_lat, input_lon, input_name)
                
                df_routes = get_dist_table(df_fallas, df_routes)

                # Un pequeño truco para contrarrestar las rutas en coche
                print('routeeeees \n', df_routes)
                df_routes = symmetric_min_routes(df_routes)

                df_matrix = create_data_matrix(df_routes)

                print('optimizing...')
                sol = optimize_tsp(df_matrix)[0]
                optimized_sequence = list(df_matrix.index[sol])
                optimized_sequence = [[optimized_sequence[i], optimized_sequence[i+1]] for i in range(len(optimized_sequence)-1)]
                
                df_routes_opt = pd.DataFrame(columns=df_routes.columns)
                for seq in optimized_sequence:
                    df_line = df_routes.loc[df_routes.ori.eq(seq[0]) & df_routes.des.eq(seq[1])]
                    df_routes_opt = pd.concat([df_routes_opt,df_line])

                df_routes_opt.reset_index(drop=True, inplace=True)
                
                # Create the map
                df_routes_opt = df_to_gdf(df_routes_opt)
                

                # Sort the fallas coordinates
                df_fallas = pd.merge(df_fallas, df_routes_opt["ori"], left_on='Falla',right_on="ori", how="right")
                df_fallas.reset_index(drop=True, inplace=True)

            except ValueError:
                st.error("Please enter valid coordinates in 'latitude,longitude' format.")
        else:
            st.warning("Please enter a valid coordinate.")

        # Crear el mapa
        map_html = create_route_map(df_fallas, df_routes_opt)  # Generar el mapa como HTML

        # Mostrar el mapa en el Streamlit
        components.html(map_html, width=1250, height=600)

# Ejecutar la función principal
if __name__ == '__main__':
    main()
