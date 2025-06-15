FROM pytorch/torchserve:latest-cpu

USER root

# Install dependencies
RUN pip install --no-cache-dir kserve torch==2.2.0 torchvision==0.17.0 pillow==10.0.0 scikit-learn==1.4.0 lime==0.2.0

# Create model store directory
RUN mkdir -p /home/model-server/model-store

# Copy model files
COPY model/ /home/model-server/model/

# Copy model archiver script and run it
COPY scripts/archive_model.sh /home/model-server/
RUN chmod +x /home/model-server/archive_model.sh
RUN /home/model-server/archive_model.sh

# Copy TorchServe configuration
COPY config/config.properties /home/model-server/config.properties

# Create and switch to non-root user
RUN chown -R model-server /home/model-server
USER model-server

# Expose ports
EXPOSE 8080 8081 8082

# Start TorchServe
CMD ["torchserve", "--start", "--ts-config=/home/model-server/config.properties", "--models", "resnet50=resnet50.mar", "--model-store", "/home/model-server/model-store"]
