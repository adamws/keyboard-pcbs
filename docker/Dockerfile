FROM admwscki/kicad-kbplacer-primary:9.0.7-noble

ARG KBPLACER_REVISION=75b2fd79a077ef56ddaa8294f4fe2cf79ef0435f

RUN apt-get update \
  && apt-get install -y nodejs npm jq zip

RUN npm install -g n && n install lts

ENV KICAD_3RDPARTY_PATH=/root/.local/share/kicad/9.0/3rdparty
ENV KICAD_CONFIG_PATH=/root/.config/kicad/9.0
ENV SWITCH_LIBRARY=com_github_perigoso_keyswitch-kicad-library
ENV WORK_PATH=/work

RUN ln -s /usr/bin/python3 /usr/bin/python

# install 3rdparty footprints
RUN mkdir -p $KICAD_3RDPARTY_PATH
RUN cd $KICAD_3RDPARTY_PATH \
  && mkdir -p footprints \
  && mkdir tmp && cd tmp \
  && wget https://github.com/kiswitch/keyswitch-kicad-library/releases/download/v2.4/keyswitch-kicad-library.zip \
  && echo "b38d56323acb91ad660567340ca938c5b4a83a27eea52308ef14aa7857b0071b keyswitch-kicad-library.zip" | sha256sum -c \
  && unzip keyswitch-kicad-library.zip \
  && rm keyswitch-kicad-library.zip \
  && mv footprints ../footprints/$SWITCH_LIBRARY \
  && cd .. && rm -rf tmp

# install color theme
RUN mkdir -p $KICAD_CONFIG_PATH/colors \
  && cd $KICAD_CONFIG_PATH/colors \
  && wget https://raw.githubusercontent.com/2231puppy/kicad-vampire/master/colors/vampire.json

RUN python -m pip install hatch
RUN git clone https://github.com/adamws/kicad-kbplacer.git \
  && cd kicad-kbplacer \
  && git checkout $KBPLACER_REVISION \
  && python -m pip install .[schematic] \
  && python -m pip install PyYAML==6.0.1 colormath==3.0.0 drawsvg==2.3.0

RUN python -m pip install yq scour==0.38.2 yq lzstring==1.0.4 Jinja2==3.1.3

COPY kbplacer-generate.sh /usr/bin

RUN mkdir -p $WORK_PATH
WORKDIR $WORK_PATH
