import cherrypy
from typing import Union
from ..production import pre_live, pre_train, osterstadt, adcarkas


class Model:
    def __init__(self, name, train, validate, live):
        self.train = train
        self.validate = validate
        self.live = live
        self.name = name


models_list = [
    Model('Osterstadt', osterstadt.train, osterstadt.validate, osterstadt.tournament),
    Model('Adcarkas', adcarkas.train, adcarkas.validate, adcarkas.tournament),
]

models = {model.name: model for model in models_list}


def run(function):
    try:
        function()
    except Exception:
        cherrypy.log(traceback=True)

def run_pre_live():
    pre_live.pre_live()

def run_osterstadt_live():
    run(osterstadt.tournament)

def run_adcarkas_live():
    run(adcarkas.tournament)


def run_pre_train():
    pre_train.pre_train()

def run_train(model_name: str):
    run_pre_train()
    run(models[model_name].train)


@cherrypy.expose
class NumeraiWebService(object):
    def __init__(self):
        self.training = NumeraiTraining()

    def _cp_dispatch(self, vpath):
        if len(vpath) == 0:
            return self
        
        if len(vpath) == 2:
            vpath.pop(0)  # /train/
            cherrypy.request.params['model_name'] = vpath.pop(0)  # /<model_name>/
            return self.training
        
        return vpath


    @cherrypy.tools.accept(media='application/json')
    @cherrypy.tools.json_in()
    def POST(self):
        cherrypy.log("Handling Numerai Webhook Request")
        data = cherrypy.request.json
        cherrypy.log(f"Data: {data}")
        if data['roundNumber'] == 'test':
            cherrypy.log("Request was test, not running script")
            return
        
        run_pre_live()
        run_osterstadt_live()
        run_adcarkas_live()


@cherrypy.expose
class NumeraiTraining(object):
    @cherrypy.tools.accept(media='text/plain')
    def POST(self, model_name):
        cherrypy.log(f"Training {model_name}")
        run_train(model_name)


if __name__ == '__main__':
    cherrypy.config.update({'server.socket_port': 3000})
    conf = {
        '/': {
            'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
        }
    }
    cherrypy.quickstart(NumeraiWebService(), '/numerai', conf)
