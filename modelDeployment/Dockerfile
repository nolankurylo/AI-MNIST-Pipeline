FROM public.ecr.aws/lambda/python:3.9

COPY requirements.txt ${LAMBDA_TASK_ROOT}
WORKDIR ${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt

COPY app.py ${LAMBDA_TASK_ROOT}
COPY basic_mnist_nn.h5 ${LAMBDA_TASK_ROOT}

#RUN file="$(ls -1 .)" && echo $file



CMD ["app.lambda_handler"]
