# FBSD
Flight Booking Status Dashboard (FBSD) is a data analysis project where from raw data (day wise booking count of flights) I tried to harness some insights and visualize them in a dashboard.

## Environments
- Python 3.11.5
- Laravel 9.26.1
- PHP 8.1.17
- mySQL 10.4.27
- echarts 5.4.3
- amcharts 5

## Dataset
Dataset is a software generated file, which contains 
- flight no
- flight date
- flight time
- route
- total capacity of the aircraft
- total seat booked

## ETL (Extract Transform Load) process
The dataset is a semi-structured tab delimited file. So, extracting data from the file will require some effort. 
### Data Extraction
Filtering the data from other texts and storing it in a dataframe 
```bash
    import pandas as pd -- import this library to perform data analysis

    generating_date = '' -- variable to store the data file receiving date
    target_string           = "target_string_to_start_data_extraction"
    generating_date_finder  = "target_string_to_get_file_receiving_date"

    df = pd.read_csv(file_path, header=None)
    
    matches             = df[df.iloc[:, 0].astype(str).str.startswith(target_string)]
    matches_target_date = df[df.iloc[:, 0].astype(str).str.startswith(generating_date_finder)]
    generating_date     = str(matches_target_date[0])[30:40]
    
    df1 = matches.iloc[:,0].str.split(expand=True)

    column_header = ['airline_code','route','flight_no','flight_date','flight_time','total_capacity','total_booked']
    df1.columns   = column_header
```
### Data Transformation
First step is to clean the data and transform the data type to the desired ones.
  ```bash
  # clearing custom flights like CRCHDQ
      for i in range(len(filtered_df)):        
          if len(filtered_df.iloc[i,1]) > 6:
              tobesplited = filtered_df.iloc[i,1]
              filtered_df.iloc[i,1], filtered_df.iloc[i,2:], filtered_df.iloc[i,2] = tobesplited[:6], filtered_df.iloc[i,2:].shift(1), tobesplited[6:]
      
      # changing data types
      d3= filtered_df.convert_dtypes()
      d3.replace('', np.nan, inplace=True)
  
      # Identify the columns intended to be integers
      int_columns = ['flight_no', 'total_capacity', 'total_booked']
      
      # Convert columns to integers and handle NaN or infinite values
      d3[int_columns] = d3[int_columns].apply(lambda x: pd.to_numeric(x, errors='coerce')).astype('Int64')
      
      d3['flight_date'] = d3['flight_date'].apply(lambda x: transform_date(x))
  
      # df_datatype_corrected = d3.astype({ 11: float, 12: float, 13:float, 14:float, 16: int, 19:float, 20:float, 21:float, 22:float})
      df_datatype_corrected = d3
  
      non_numeric_columns = df_datatype_corrected.select_dtypes(exclude=['number', 'datetime']).columns
      df_datatype_corrected[non_numeric_columns] = df_datatype_corrected[non_numeric_columns].fillna('N/A')
      
      df_nullCellTransformed = df_datatype_corrected.fillna(0)
      df_allCellTrimmed = trim_all_columns(df_nullCellTransformed)
  
  ```
* Trimming all columns
  
  ```
  def trim_all_columns(df):
      """
      Trim whitespace from ends of each value across all series in dataframe
      """
      trim_strings = lambda x: x.strip() if isinstance(x, str) else x
      return df.applymap(trim_strings)
  ```

* Transform date to desired format

  ```
  def transform_date(input_date):
      formats = ['%m/%d/%Y', '%m/%d/%y', '%d/%m/%Y', '%d/%m/%y', "%b'%y"]
      for date_format in formats:
          try:
              date_obj = datetime.strptime(input_date, date_format)
              return date_obj.strftime('%Y-%m-%d')
          except ValueError:
              continue
      raise ValueError('Invalid date format: {}'.format(input_date))
  ```

* Calculate day in the week

  ```
  def get_day(input_date):
    formats = ['%m/%d/%Y', '%m/%d/%y', '%d/%m/%Y', '%d/%m/%y', "%b'%y", '%Y-%m-%d']
    for date_format in formats:
        try:
            date_obj = datetime.strptime(input_date, date_format)
            day_of_week = date_obj.weekday()
            return day_of_week
        except ValueError:
            continue
    raise ValueError('Invalid date format: {}'.format(input_date))
  ```

* Get Geo-Location of all airports to calculate the distance between them*
  ```
  def get_all_geo_location():
      print("---------START: Reading All_Geo_Location Data---------")
      
      import pymysql
      global df_geo_location
      
      mydb_connection = pymysql.connect(
          host = "host_name",
          user = "user_name",
          database = "db_name",
          password = "password")
      
      mycursor = mydb_connection.cursor()
      
      # Execute the first query to retrieve all latitude and longitude from tha database
      origin_query = "SELECT iata_code, latitude, longitude FROM airport_details"
      mycursor.execute(origin_query)    
      df_geo_location = pd.DataFrame(mycursor.fetchall())
      
      # Close the cursor and the connection
      mycursor.close()
      mydb_connection.close()
      
      column_header = ['iata_code','latitude','longitude']
      df_geo_location.columns = column_header
      
      df_geo_location = df_geo_location[df_geo_location['iata_code'] != r'\N']
      
      df_geo_location['latitude'] = df_geo_location['latitude'].astype(float)
      df_geo_location['longitude'] = df_geo_location['longitude'].astype(float)
      print("---------END: Reading All_Geo_location Data---------")
  
      return df_geo_location
  ```

* Now use Transformed dataframe and all airports geo-loation to calculate RPK and ASK *

  ```
      print("---------Transforming starting for "+file+"----------")
      global df_merged
      
      cleaned_df['origin']        = cleaned_df['route'].apply(lambda x: x[:3])
      cleaned_df['destination']   = cleaned_df['route'].apply(lambda x : x[3:])
      cleaned_df['day_of_week']   = cleaned_df['flight_date'].apply(lambda x: get_day(x))
      cleaned_df['generated_at']  = transform_date(generating_date)
     
      
      # assume that cleaned_df is your large dataset and all_location is the latitude and longitude containing dataframe
      df_merged = pd.merge(cleaned_df, all_location, left_on='origin', right_on='iata_code', how='left')
      df_merged = pd.merge(df_merged, all_location, left_on='destination', right_on='iata_code', how='left', suffixes=('_origin', '_destination'))
  
  
      # assume that df_merged is your dataframe with columns latitude_origin, longitude_origin, latitude_destination, longitude_destination
      df_merged['distance_km'] = calculate_distance_vec(df_merged['latitude_origin'], df_merged['longitude_origin'], df_merged['latitude_destination'], df_merged['longitude_destination'])
      
      df_merged['distance_km'] = pd.to_numeric(df_merged['distance_km'], errors='coerce')
      df_merged['rpk']         = df_merged['tot_bkd'] * df_merged['distance_km']
      
      
      # Columns to exclude
      columns_to_exclude = ['iata_code_origin', 'latitude_origin', 'longitude_origin', 'iata_code_destination', 'latitude_destination', 'longitude_destination']
      
      # Using drop() method
      df_merged_copy_1   = df_merged.drop(columns=columns_to_exclude, inplace=False)
  
  
      print("---------Transforming ending for "+file+"----------")
  ```

* Way to calculate distance from latitude and longitude of two points

  ```
  def calculate_distance_vec(lat1, lon1, lat2, lon2):
  
      import numpy as np
  
      radius_earth = 6.371E3  # km
      phi1         = np.radians(lat1)
      phi2         = np.radians(lat2)
      delta_phi    = np.radians(lat1 - lat2)
      delta_lam    = np.radians(lon1 - lon2)
  
      a = np.sin(0.5 * delta_phi)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(0.5 * delta_lam)**2
      c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
  
      distance_km = radius_earth * c
      return distance_km
  ```

* To recognize whether the route is a segment of previous route or a new flight

  ```
  def accumulate_segment(row, temp_storage):
    flight_no = row['flight_no']
    flight_date = row['flight_date']
    origin = row['origin']
    destination = row['destination']
    
    if flight_no == temp_storage['flight_no']:
        if flight_date == temp_storage['flight_date']:
            if temp_storage['origin2'] and temp_storage['destination2']:
                temp_storage['journey_date'] = flight_date
                temp_storage['flight_no'] = flight_no
                temp_storage['flight_date'] = flight_date
                temp_storage['origin1'] = ''
                temp_storage['destination1'] = ''
                temp_storage['origin2'] = ''
                temp_storage['destination2'] = ''
            else:
                temp_storage['journey_date'] = flight_date
                temp_storage['origin2'] = origin
                temp_storage['destination2'] = destination
                temp_storage['flight_no'] = flight_no
                temp_storage['flight_date'] = flight_date
        else:
            if ((pd.to_datetime(flight_date) - pd.to_datetime(temp_storage['journey_date'])) == pd.Timedelta(days=1)):
                if temp_storage['origin1'] and temp_storage['destination1']:
                    if temp_storage['origin2'] and temp_storage['destination2']:
                        if ((origin == temp_storage['destination1'] and destination == temp_storage['destination2']) or (destination == temp_storage['destination1'] and origin == temp_storage['destination2'])):
                            temp_storage['journey_date'] = temp_storage['journey_date']
                            temp_storage['flight_no'] = flight_no
                            temp_storage['flight_date'] = flight_date
                            temp_storage['origin1'] = origin
                            temp_storage['destination1'] = destination
                            temp_storage['origin2'] = ''
                            temp_storage['destination2'] = ''
                        else:
                            temp_storage['journey_date'] = flight_date
                            temp_storage['flight_no'] = flight_no
                            temp_storage['flight_date'] = flight_date
                            temp_storage['origin1'] = origin
                            temp_storage['destination1'] = destination
                            temp_storage['origin2'] = ''
                            temp_storage['destination2'] = ''
                    else:
                        if origin == temp_storage['origin1'] and destination == temp_storage['destination1']:
                            temp_storage['journey_date'] = flight_date
                            temp_storage['origin1'] = origin
                            temp_storage['destination1'] = destination
                            temp_storage['origin2'] = ''
                            temp_storage['destination2'] = ''
                            temp_storage['flight_no'] = flight_no
                            temp_storage['flight_date'] = flight_date
                        elif origin == temp_storage['origin1'] and destination != temp_storage['destination1']: 
                            temp_storage['journey_date'] = temp_storage['journey_date']
                            temp_storage['flight_no'] = flight_no
                            temp_storage['flight_date'] = temp_storage['flight_date']
                            temp_storage['origin2'] = origin
                            temp_storage['destination2'] = destination
                        elif origin == temp_storage['destination1'] and destination != temp_storage['origin1']: 
                            temp_storage['journey_date'] = temp_storage['journey_date']
                            temp_storage['flight_no'] = flight_no
                            temp_storage['flight_date'] = temp_storage['flight_date']
                            temp_storage['origin2'] = origin
                            temp_storage['destination2'] = destination
                        elif destination == temp_storage['destination1'] and origin != temp_storage['origin1']: 
                            temp_storage['journey_date'] = temp_storage['journey_date']
                            temp_storage['flight_no'] = flight_no
                            temp_storage['flight_date'] = temp_storage['flight_date']
                            temp_storage['origin2'] = origin
                            temp_storage['destination2'] = destination
                else:
                    temp_storage['journey_date'] = flight_date
                    temp_storage['origin1'] = origin
                    temp_storage['destination1'] = destination
                    temp_storage['origin2'] = ''
                    temp_storage['destination2'] = ''
                    temp_storage['flight_no'] = flight_no
                    temp_storage['flight_date'] = flight_date
                    
            else:
                temp_storage['journey_date'] = flight_date
                temp_storage['origin1'] = origin
                temp_storage['destination1'] = destination
                temp_storage['origin2'] = ''
                temp_storage['destination2'] = ''
                temp_storage['flight_no'] = flight_no
                temp_storage['flight_date'] = flight_date
                
    else:
        temp_storage['journey_date'] = flight_date
        temp_storage['origin1'] = origin
        temp_storage['destination1'] = destination
        temp_storage['origin2'] = ''
        temp_storage['destination2'] = ''
        temp_storage['flight_no'] = flight_no
        temp_storage['flight_date'] = flight_date
    
    return (temp_storage['journey_date'], temp_storage['flight_no'],temp_storage['flight_date'])
  ```

* Now sort the dataframe with respect to flight_no, flight_date, route and generated_at and then call the accumulate_Segmaent function to find out actual flight route
 
  ```
   future_flights_grouped = df.sort_values(by=['flight_no', 'flight_date', 'route', 'generated_at'])
    
   temp_storage = {'origin1': '', 'destination1': '', 'flight_no': '','origin2': '', 'destination2': '', 'flight_date': '', 'journey_date': ''}

   results = future_flights_grouped.apply(lambda row: accumulate_segment(row, temp_storage), axis=1)
   future_flights_grouped['journey_date'] = results.apply(lambda x: x[0])
  ```


* Now calculate the ASK (Available Seat per Kilometer)
 
  ```
  def calculate_ask_distance(row, temp_storage):
          origin = row['origin']
          destination = row['destination']
          distance = row['distance_km']
    
        if temp_storage['origin1'] and temp_storage['destination1']:
            if temp_storage['origin2'] and temp_storage['destination2']:
                if origin == temp_storage['destination1']:
                    temp_storage['total_distance'] = distance + temp_storage['distance1']
                    temp_storage['route'] = temp_storage['origin1'] + '-' + temp_storage['destination1'] + '-' + destination
                elif origin == temp_storage['destination2']:
                    temp_storage['total_distance'] = distance + temp_storage['distance2']
                    temp_storage['route'] = temp_storage['origin2'] + '-' + temp_storage['destination2'] + '-' + destination
                elif origin == temp_storage['origin1'] and destination == temp_storage['origin2']:
                    temp_storage['total_distance'] += distance 
                    temp_storage['route'] = temp_storage['origin1'] + '-' + temp_storage['origin2'] + '-' + temp_storage['destination2']
                elif origin == temp_storage['origin2'] and destination == temp_storage['origin1']:
                    temp_storage['total_distance'] += distance 
                    temp_storage['route'] = temp_storage['origin2'] + '-' + temp_storage['origin1'] + '-' + temp_storage['destination1']
                    
            else:
                if origin == temp_storage['origin1'] or destination == temp_storage['destination1']:
                    temp_storage['origin2'] = origin
                    temp_storage['destination2'] = destination
                    temp_storage['distance2'] = distance  
                elif origin == temp_storage['destination1']:
                    temp_storage['origin2'] = origin
                    temp_storage['destination2'] = destination
                    temp_storage['distance2'] = distance
                    temp_storage['total_distance'] += distance
                    temp_storage['route'] = temp_storage['origin1'] + '-' + temp_storage['destination1'] + '-' + destination
                    # print("this route is transit point")
                elif destination == temp_storage['origin1']:
                    temp_storage['origin2'] = origin
                    temp_storage['destination2'] = destination
                    temp_storage['distance2'] = distance
                    temp_storage['total_distance'] += distance
                    temp_storage['route'] = origin +  '-' + destination + '-' + temp_storage['destination1'] 
                    # print("this route is transit point")
        else:
            temp_storage['origin1'] = origin
            temp_storage['destination1'] = destination
            temp_storage['distance1'] = distance
            temp_storage['total_distance'] = distance
            temp_storage['route'] = origin +  '-' + destination
            # print(row['flight_no'], row['flight_date'] ,origin, destination)
            
        # print(origin,destination,distance,temp_storage['total_distance'])
    
    
        return (temp_storage['total_distance'],temp_storage['route'])
  ```

* Finding out PLF (Passenger Load Factor)

  ```
      def calculate_plf(group):
          temp_storage = {'origin1': '', 'destination1': '', 'distance1': 0,'origin2': '', 'destination2': '', 'distance2': 0, 'total_distance': 0, 'route': ''}
      
          results = group.apply(lambda row: calculate_ask_distance(row, temp_storage), axis=1)
          group['total_distance'] = results.apply(lambda x: x[0])
          group['route'] = results.apply(lambda x: x[1])
      
          flight_no = group['flight_no'].iloc[0]
          journey_date = group['journey_date'].iloc[0]
          generated_at = group['generated_at'].iloc[0]
          total_RPK = group['rpk'].sum()
          total_booked = group['booked_total'].sum()
          total_seat = group['capacity_total'].iloc[0]
          
          total_distance = group['total_distance'].iloc[-1]  # Get the last accumulated distance
          total_route = group['route'].iloc[-1]  # Get the last route
          plf = (total_RPK / (total_seat * total_distance)) * 100 if total_seat * total_distance != 0 else 0
      
          return pd.Series({
              'flight_no': flight_no, 
              'route': total_route,
              'flight_date': journey_date,
              'total_seat': total_seat, 
              'total_booked': total_booked, 
              'total_distance': total_distance,
              'total_RPK': total_RPK, 
              'total_ASK': total_seat * total_distance, 
              'plf': plf,
              'generated_at': generated_at
          })
  ```

* Call the above functions using below line

  ```
   future_flights_grouped = future_flights_grouped.groupby(['flight_no', 'journey_date']).apply(calculate_plf)
  ```


* To store the summary into database
  ```
  
      engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                             .format(user="user_name",
                                     host="host_name",
                                     pw="password",
                                     db="db_name"))
      df_data_reset = df_data_final.reset_index(drop=True)
      
      # Define a mapping from Pandas data types to SQL data types and default values
  
      pandas_to_sql_type_mapping = {
          'int32': {'type': 'INT', 'default': '0'},
          'int64': {'type': 'INT', 'default': '0'},
          'float64': {'type': 'FLOAT', 'default': '0.0'},
          'object': {'type': 'TEXT', 'default': "''"},  # Default empty string for text
          'datetime64[ns]': {'type': 'TIMESTAMP', 'default': 'CURRENT_TIMESTAMP'},  # Example of using a function as a default
      }
  
      
      SQL_CREATE_TBL = "CREATE TABLE IF NOT EXISTS flight_status(id INT NOT NULL AUTO_INCREMENT,"
      
      for column_name, data_type in zip(df_data_reset.columns, df_data_reset.dtypes):
          # Retrieve the mapping for the current data type
          sql_type_info = pandas_to_sql_type_mapping.get(str(data_type), {'type': 'TEXT', 'default': "''"})
          sql_data_type = sql_type_info['type']
          default_value = sql_type_info['default']
      
          # Add column definition to the SQL statement with default values
          SQL_CREATE_TBL += "{} {} DEFAULT {}, ".format(column_name, sql_data_type, default_value)
  
      SQL_CREATE_TBL += "created_at TIMESTAMP NOT NULL DEFAULT current_timestamp(), updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp(), PRIMARY KEY (id));"
      
      mycursor = mydb.cursor()
      
      def insert_ignore(table, conn, keys, data_iter):
          from sqlalchemy.ext.compiler import compiles
          from sqlalchemy.sql.expression import Insert
      
          @compiles(Insert)
          def replace_string(insert, compiler, **kw):
              s = compiler.visit_insert(insert, **kw)
              s = s.replace("INSERT INTO", "INSERT IGNORE INTO")
              return s
      
          data = [dict(zip(keys, row)) for row in data_iter]
          conn.execute(table.table.insert(), data)
  
      
      try:
          print("Creating table {}: ".format("flight_status"), end='')
          mycursor.execute(SQL_CREATE_TBL)
          df_data_reset.to_sql('flight_status', con=engine, if_exists='append', index=False, method=insert_ignore)
      except pymysql.Error as err:
          print(err)
          pass
      else:
          print("OK")
  ```

* To automate the process, so that upon running the script the system will check for desired file and start ETL automatically

  ```
  import os
  
  uploadedFolder = r"folder_location_to_get_the_file_to_process"
  insertedFolder = r"folder_location_to_store_the_processed_file"
  
  for file in os.listdir(uploadedFolder):
      if file.endswith('.TXT'):
          file_path       = uploadedFolder +"\\"+ file
          read_df         = read_file(file_path)
  ```

* To save the processed data as a file

  ```
  def relocate_file(file, file_path, processedFileFolder, transformed_df):
    print("---------Renaming and Relocating starting for "+file+"----------")
    
    timestr = time.strftime("%Y%m%d_%H%M%S_")
    transformedDataFile =  timestr + transformed_df.iloc[0].generated_at + "_"+transformed_df['flight_date'].iloc[0]+"_"+transformed_df['flight_date'].iloc[-1]+".xlsx"
    uploaded_file_path  = processedFileFolder + "\\" + transformedDataFile
    transformed_df.to_excel(uploaded_file_path, index= False)
    
    backup_file_path = processedFileFolder + "\\" + timestr + file
    newName = pathlib.PurePosixPath(backup_file_path).stem + '.TXT'
    
    os.rename(file_path, newName)
    
    print("---------Renaming and Relocating ended for "+file+"----------")
  ```
