from kafka import KafkaProducer, KafkaConsumer
import json


class Converter:
    def __init__(self):
        self.sequence = []
        self.consumer = KafkaConsumer(
                'raw-data',
                bootstrap_servers=['localhost:9092'],
                group_id='python_converter',
                auto_offset_reset='earliest',
                value_deserializer=lambda v: json.loads(v.decode('utf-8'))
                )
        self.producer = KafkaProducer(
                bootstrap_servers=['localhost:9092'],
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
                )

        for mssg in self.consumer:
            self.convert_data(mssg)

    def convert_data(self, data):
        pass


