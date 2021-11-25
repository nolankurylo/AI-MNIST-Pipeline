FROM public.ecr.aws/lambda/python:3.9

COPY app.py requirements.txt ./
COPY model /opt/ml/model

RUN pip install -r requirements.txt

CMD ["app.lambda_handler"]
