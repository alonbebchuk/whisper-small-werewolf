
echo "installing gcsfuse"
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc

# curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
while true; do
    if command -v python3.10 &> /dev/null; then
        echo "Python 3.10 is already installed."
        break
    else
        echo "Python 3.10 is not installed. Installing..."
        yes | sudo add-apt-repository ppa:deadsnakes/ppa
        sudo apt update
        if sudo apt install -y python3.10 python3.10-distutils python3.10-dev gcsfuse jq redis-tools; then
            echo "Python 3.10 has been successfully installed."
            break
        else
            sudo pkill -9 unattended-upgr && sleep 2 && sudo pkill -9 apt
            # sudo rm -rf /var/lib/dpkg/info/snapd.* && sudo dpkg --remove --force-remove-reinstreq snapd && sudo apt-get update && sudo apt-get -f install && sudo dpkg --configure -a
            sleep 2
            sudo rm /var/{lib/{apt/lists,dpkg}/lock,lib/dpkg/lock-frontend,cache/apt/archives/lock}
            sleep 2
            sudo dpkg --configure -a
            echo "Failed to install Python 3.10. Retrying..."
            sleep 5
        fi
    fi
done
wget https://bootstrap.pypa.io/get-pip.py
python3.10 get-pip.py
sudo apt-get -o DPkg::Lock::Timeout=-1 update


python3.10 -m pip install apache-beam==2.51.0
python3.10 -m pip install setuptools==65.7.0 gradio
python3.10 -m pip install --use-pep517 -r requirements.txt


# old jax, old tensorflow, old numpy
python3.10 -m pip install --upgrade --force-reinstall numpy~=1.0 "jax[tpu]==0.4.33" flax==0.8.5 optax==0.2.2 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html tensorboard-plugin-profile==2.8.0 tensorboard-plugin-wit==1.8.1 tensorflow-datasets==4.6.0 tensorflow-estimator==2.9.0 tensorflow-hub==0.12.0 tensorflow-io-gcs-filesystem==0.26.0 tensorflow-io==0.26.0 tensorflow-metadata==1.12.0 tensorflow-probability==0.17.0 tensorflow-text==2.9.0 tensorflow==2.9.1 googleapis-common-protos==1.58.0 proto-plus==1.22.2 protobuf==3.19.6 
python3.10 -m pip install --upgrade --force-reinstall protobuf==3.19.6 

# python3.10 -m pip install --upgrade --force-reinstall numpy~=1.0 "jax[tpu]==0.4.33" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
