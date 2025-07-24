FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV MODEL_FILE=checkpoint_epoch_50.pth

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app/ .

COPY startup.sh .
RUN chmod +x startup.sh

EXPOSE 5000

CMD ["./startup.sh"]
