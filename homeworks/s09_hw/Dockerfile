FROM agrigorev/zoomcamp-bees-wasps:v2

RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl
#RUN pip install scipy
RUN pip install Pillow

COPY lambda_function.py .

CMD [ "lambda_function.lambda_handler" ]
