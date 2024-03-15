import tables

from base.config_loader import ConfigLoader

COLUMN_MOUSE_ID = 'mouse_id'
COLUMN_LABEL = 'label'
COLUMN_LAB = 'lab'

def create_table_description(config: ConfigLoader):
    """ creates the description for the pytables table used for dataloading """
    n_sample_values = int(config.SAMPLING_RATE * config.SAMPLE_DURATION)

    table_description = {
        COLUMN_MOUSE_ID: tables.StringCol(50),
        COLUMN_LABEL: tables.StringCol(10),
        COLUMN_LAB: tables.StringCol(10)
    }
    for c in config.CHANNELS_IN_TABLE:
        table_description[c] = tables.Float32Col(shape=n_sample_values)

    return table_description
