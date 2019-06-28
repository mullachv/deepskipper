FROM ubuntu:16.04

RUN apt-get update
RUN echo 'export PATH=~/.local/bin:$PATH\n' >> $HOME/.bashrc

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
RUN rm ./requirements.txt

COPY /usr/local/bin/ffmpeg                              /usr/local/bin/
COPY /Users/vmullachery/mine/insight/2019060200             ./
COPY /Users/vmullachery/mine/insight/2018060700             ./
COPY /Users/vmullachery/mine/insight/gym-shipping           ./
COPY /Users/vmullachery/mine/insight/nautlabs               ./
COPY /Users/vmullachery/mine/insight/pgrad_agent.py         ./
COPY /Users/vmullachery/mine/insight/map_animate.ipynb      ./
COPY /Users/vmullachery/mine/insight/runp.sh                ./

RUN sed -i '/basedatafolder/s|/Users/vmullachery/mine/insight/|./|g' nautlabs/shipperf.py

# train and save model, output csv file
RUN nohup ./runp.sh > ./outputs/pgradrun.out 2>&1 &
