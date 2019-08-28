import json
import math

from substra.sdk import assets


def find_dict_composite_key_value(asset_dict, composite_key):
    def _recursive_find(d, keys):
        value = d.get(keys[0])
        if len(keys) == 1:
            return value
        return _recursive_find(value or {}, keys[1:])

    return _recursive_find(asset_dict, composite_key.split('.'))


class Field:
    def __init__(self, name, ref):
        self.name = name
        self.ref = ref

    def get_value(self, item, expand=False):
        return find_dict_composite_key_value(item, self.ref)

    def print_single(self, item, field_length, expand):
        name = self.name.upper().ljust(field_length)
        value = self.get_value(item, expand)

        if isinstance(value, list):
            if value:
                print(name, end='')
                padding = ' ' * field_length
                for i, v in enumerate(value):
                    if i == 0:
                        print(f'- {v}')
                    else:
                        print(f'{padding}- {v}')
            else:
                print(f'{name} None')
        else:
            print(f'{name}{value}')


class PermissionField(Field):
    def get_value(self, item, expand=False):
        value = super().get_value(item, expand)
        return 'owner only' if value == [] else value


class DataSampleKeysField(Field):
    def get_value(self, item, expand=False):
        value = super().get_value(item, expand)
        if not expand and value:
            n = len(value)
            value = f'{n} data sample key' if n == 1 else f'{n} data sample keys'
        return value


class BasePrinter:
    asset_name = None

    key_field = Field('key', 'key')
    many_fields = ()
    single_fields = ()

    download_message = None
    has_description = True

    def print_list(self, items, raw):
        """Display many items."""

        if raw:
            print(json.dumps(items, indent=2))
            return

        columns = []
        for field in self._get_many_fields():
            values = [str(field.get_value(item)) for item in items]

            column = [field.name.upper()]
            column.extend(values)

            columns.append(column)

        column_widths = []
        for column in columns:
            width = max([len(x) for x in column])
            width = (math.ceil(width / 4) + 1) * 4
            column_widths.append(width)

        for row_index in range(len(items) + 1):
            for col_index, column in enumerate(columns):
                print(column[row_index].ljust(column_widths[col_index]), end='')
            print()

    def _get_many_fields(self):
        return (self.key_field, ) + self.many_fields

    def _get_single_fields(self):
        return (self.key_field, ) + self.single_fields

    def _get_asset_field_length(self):
        fields = [field.name for field in self._get_single_fields()]
        max_field_length = max([len(x) for x in fields])
        field_length = (math.ceil(max_field_length / 4) + 1) * 4
        return field_length

    def print_download_message(self, item):
        if self.download_message:
            key_value = self.key_field.get_value(item)
            print()
            print(self.download_message)
            print(f'\tsubstra download {self.asset_name} {key_value}')

    def print_description_message(self, item):
        if self.has_description:
            key_value = self.key_field.get_value(item)
            print()
            print(f'Display this {self.asset_name}\'s description:')
            print(f'\tsubstra describe {self.asset_name} {key_value}')

    def print_messages(self, item):
        self.print_download_message(item)
        self.print_description_message(item)

    def print_single(self, item, raw, expand):
        """Display single item."""

        if raw:
            print(json.dumps(item, indent=2))
            return

        field_length = self._get_asset_field_length()
        for field in self._get_single_fields():
            field.print_single(item, field_length, expand)

        self.print_messages(item)


class JsonOnlyPrinter:
    @staticmethod
    def _print(data):
        print(json.dumps(data, indent=2))

    def print_list(self, items, raw):
        self._print(items)

    def print_single(self, item, raw):
        self._print(item)


class AlgoPrinter(BasePrinter):
    asset_name = 'algo'

    many_fields = (
        Field('Name', 'name'),
    )
    single_fields = (
        Field('Name', 'name'),
        PermissionField('Permissions', 'permissions'),
    )

    download_message = 'Download this algorithm\'s code:'


class ObjectivePrinter(BasePrinter):
    asset_name = 'objective'

    many_fields = (
        Field('Name', 'name'),
        Field('Metrics', 'metrics.name'),
    )
    single_fields = (
        Field('Name', 'name'),
        Field('Metrics', 'metrics.name'),
        Field('Test dataset key', 'testDataset.dataManagerKey'),
        DataSampleKeysField('Test data sample keys', 'testDataset.dataSampleKeys'),
        PermissionField('Permissions', 'permissions'),
    )
    download_message = 'Download this objective\'s metric:'

    def print_leaderboard_message(self, item):
        key_value = self.key_field.get_value(item)
        print()
        print('Display this objective\'s leaderboard:')
        print(f'\tsubstra leaderboard {key_value}')

    def print_messages(self, item):
        super().print_messages(item)
        self.print_leaderboard_message(item)


class DataSamplePrinter(BasePrinter):
    asset_name = 'data sample'


class DatasetPrinter(BasePrinter):
    asset_name = 'dataset'

    many_fields = (
        Field('Name', 'name'),
        Field('Type', 'type'),
    )
    single_fields = (
        Field('Name', 'name'),
        Field('Objective key', 'objectiveKey'),
        Field('Type', 'type'),
        DataSampleKeysField('Train data sample keys', 'trainDataSampleKeys'),
        DataSampleKeysField('Test data sample keys', 'testDataSampleKeys'),
        PermissionField('Permissions', 'permissions'),
    )
    download_message = 'Download this data manager\'s opener:'


class TraintuplePrinter(BasePrinter):
    asset_name = 'traintuple'

    many_fields = (
        Field('Algo name', 'algo.name'),
        Field('Status', 'status'),
        Field('Perf', 'dataset.perf'),
    )
    single_fields = (
        Field('Model key', 'outModel.hash'),
        Field('Algo key', 'algo.hash'),
        Field('Algo name', 'algo.name'),
        Field('Objective key', 'objective.hash'),
        Field('Status', 'status'),
        Field('Perf', 'dataset.perf'),
        DataSampleKeysField('Train data sample keys', 'dataset.keys'),
        Field('Rank', 'rank'),
        Field('Compute Plan Id', 'computePlanID'),
        Field('Tag', 'tag'),
        Field('Log', 'log'),
        PermissionField('Permissions', 'permissions'),
    )
    has_description = False


class TesttuplePrinter(BasePrinter):
    asset_name = 'testtuple'

    many_fields = (
        Field('Algo name', 'algo.name'),
        Field('Certified', 'certified'),
        Field('Status', 'status'),
        Field('Perf', 'dataset.perf')
    )
    single_fields = (
        Field('Traintuple key', 'model.traintupleKey'),
        Field('Algo key', 'algo.hash'),
        Field('Algo name', 'algo.name'),
        Field('Objective key', 'objective.hash'),
        Field('Certified', 'certified'),
        Field('Status', 'status'),
        Field('Perf', 'dataset.perf'),
        DataSampleKeysField('Test data sample keys', 'dataset.keys'),
        Field('Tag', 'tag'),
        Field('Log', 'log'),
        PermissionField('Permissions', 'permissions'),
    )
    has_description = False


PRINTERS = {
    assets.ALGO: AlgoPrinter,
    assets.OBJECTIVE: ObjectivePrinter,
    assets.DATASET: DatasetPrinter,
    assets.DATA_SAMPLE: DataSamplePrinter,
    assets.TRAINTUPLE: TraintuplePrinter,
    assets.TESTTUPLE: TesttuplePrinter,
}


def get_printer(asset):
    return PRINTERS[asset]() if asset in PRINTERS else JsonOnlyPrinter()
