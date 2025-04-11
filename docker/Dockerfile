FROM admwscki/kicad-kbplacer-primary:8.0.9-jammy

ARG KBPLACER_REVISION=v0.13

RUN apt-get update \
  && apt-get install -y jq x11-apps x11-utils x11-xserver-utils xdotool zip

ENV XAUTHORITY=/root/.Xauthority
ENV KICAD_3RDPARTY_PATH=/root/.local/share/kicad/8.0/3rdparty
ENV KICAD_TEMPLATE_PATH=/root/.local/share/kicad/8.0/template
ENV KICAD_CONFIG_PATH=/root/.config/kicad/8.0
ENV SWITCH_LIBRARY=com_github_perigoso_keyswitch-kicad-library
ENV WORK_PATH=/work

RUN touch $XAUTHORITY && chmod 600 $XAUTHORITY

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

RUN mkdir -p $KICAD_TEMPLATE_PATH
COPY template.kicad_pro $KICAD_TEMPLATE_PATH

RUN python -m pip install hatch
RUN git clone https://github.com/adamws/kicad-kbplacer.git \
  && cd kicad-kbplacer \
  && git checkout $KBPLACER_REVISION \
  && python -m pip install . \
  && python -m pip install PyYAML==6.0.1 colormath==3.0.0 drawsvg==2.3.0 kicad-skip==0.2.5

RUN python -m pip install yq scour==0.38.2 yq pyurlon==0.1.0 Jinja2==3.1.3
RUN python -m pip install git+https://github.com/adamws/kinet2pcb@1fc8a4b67e9fbb25beade950c80d3cdc8d868c23

COPY eeschema-open-and-save.sh /usr/bin
COPY kbplacer-generate.sh /usr/bin

RUN mkdir -p $WORK_PATH
WORKDIR $WORK_PATH
