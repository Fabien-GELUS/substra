from unittest import mock

from .test_base import data_manager, objective
from .test_base import TestBase, mock_success_response, mock_fail_response


def mocked_requests_get_objective(*args, **kwargs):
    return mock_success_response(data=objective)


def mocked_requests_get_data_manager(*args, **kwargs):
    return mock_success_response(data=data_manager)


def mocked_requests_get_objective_fail(*args, **kwargs):
    return mock_fail_response()


class TestGet(TestBase):

    @mock.patch('substra.sdk.requests_wrapper.requests.get', side_effect=mocked_requests_get_objective)
    def test_returns_objective_list(self, mock_get):
        res = self.client.get(
            'objective',
            'd5002e1cd50bd5de5341df8a7b7d11b6437154b3b08f531c9b8f93889855c66f')

        self.assertEqual(res, objective)
        self.assertEqual(len(mock_get.call_args_list), 1)

    @mock.patch('substra.sdk.requests_wrapper.requests.get', side_effect=mocked_requests_get_objective_fail)
    def test_returns_objective_list_fail(self, mock_get):
        try:
            self.client.get(
                'objective',
                'd5002e1cd50bd5de5341df8a7b7d11b6437154b3b08f531c9b8f93889855c66f')
        except Exception as e:
            print(str(e))
            self.assertEqual(str(e), '500')

        self.assertEqual(len(mock_get.call_args_list), 1)

    @mock.patch('substra.sdk.requests_wrapper.requests.get', side_effect=mocked_requests_get_data_manager)
    def test_returns_data_manager_list(self, mock_get):
        res = self.client.get(
            'data_manager',
            'ccbaa3372bc74bce39ce3b138f558b3a7558958ef2f244576e18ed75b0cea994')

        self.assertEqual(res, data_manager)
        self.assertEqual(len(mock_get.call_args_list), 1)