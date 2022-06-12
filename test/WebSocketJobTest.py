import unittest
from jobs.WebSocketJob import WebSocketJob
import pandas as pd


class WebSocketJobTest(unittest.TestCase):

    def test_upper(self):
        def callback(result):
            print("Callback: " + str(result))

        websocketJob = WebSocketJob()
        websocketJob.addListener(callback)
        websocketJob.start()

        self.assertEqual('foo'.upper(), 'FOO')

    # def saveToFileCSV(self, data):
    #     file = "./static/data/tuan.csv"
    #     df = pd.DataFrame(data)
    #     df.to_csv(file, index=False)


# def test_isupper(self):
#     self.assertTrue('FOO'.isupper())
#     self.assertFalse('Foo'.isupper())

# def test_split(self):
#     s = 'hello world'
#     self.assertEqual(s.split(), ['hello', 'world'])
#     # check that s.split fails when the separator is not a string
#     with self.assertRaises(TypeError):
#         s.split(2)


if __name__ == '__main__':
    unittest.main()
