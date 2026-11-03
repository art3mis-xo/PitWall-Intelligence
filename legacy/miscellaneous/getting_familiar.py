import fastf1

# session = fastf1.get_session(2021, 7, 'Q')
# print(session.name)

# session = fastf1.get_session(2021, 7, 'Q')

session = fastf1.get_session(2021, 'Las Vegas', 'Race')
# print(session.name)
session.load()
session.results
# fastf1.get_event_schedule(2021, 'Spanish Grand Prix')
# fastf1.get_events_remaining(2021)
# fastf1.get_session(2021, 'Spanish Grand Prix', 'R').load()
# fastf1.get_session(2021, 'Spanish Grand Prix', 'R').results
# fastf1.get_session(2021, 'Spanish Grand Prix', 'R').laps
# fastf1.get_session(2021, 'Spanish Grand Prix', 'R').drivers
# fastf1.get_session(2021, 'Spanish Grand Prix', 'R').teams   
# fastf1.get_session(2021, 'Spanish Grand Prix', 'R').weather
# fastf1.get_session(2021, 'Spanish Grand Prix', 'R').track_status    
# fastf1.get_session(2021, 'Spanish Grand Prix', 'R').track_status_data
# fastf1.get_session(2021, 'Spanish Grand Prix', 'R').track_status_data['LapNumber']
# fastf1.get_session(2021, 'Spanish Grand Prix', 'R').track_status_data['TrackStatus']