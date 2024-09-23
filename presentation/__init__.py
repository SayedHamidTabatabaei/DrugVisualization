from flask import Flask
from injector import Injector, Binder

from businesses.drug_business import DrugBusiness
from businesses.drug_embedding_business import DrugEmbeddingBusiness
from businesses.enzyme_business import EnzymeBusiness
from businesses.job_business import JobBusiness
from businesses.pathway_business import PathwayBusiness
from businesses.reduction_business import ReductionBusiness
from businesses.similarity_business import SimilarityBusiness
from businesses.target_business import TargetBusiness
from presentation.controllers.drug_controller import DrugController
from presentation.controllers.drug_embedding_controller import DrugEmbeddingController
from presentation.controllers.enzyme_controller import EnzymeController
from presentation.controllers.job_controller import JobController
from presentation.controllers.pathway_controller import PathwayController
from presentation.controllers.reduction_controller import ReductionController
from presentation.controllers.similarity_controller import SimilarityController
from presentation.controllers.target_controller import TargetController
from presentation.controllers.training_controller import TrainingController

app = Flask(__name__)
app.config['TIMEOUT'] = 2


def configure(binder: Binder) -> None:
    binder.bind(DrugBusiness, to=DrugBusiness)
    binder.bind(EnzymeBusiness, to=EnzymeBusiness)
    binder.bind(PathwayBusiness, to=PathwayBusiness)
    binder.bind(TargetBusiness, to=TargetBusiness)
    binder.bind(SimilarityBusiness, to=SimilarityBusiness)
    binder.bind(DrugEmbeddingBusiness, to=DrugEmbeddingBusiness)
    binder.bind(ReductionBusiness, to=ReductionBusiness)
    binder.bind(JobBusiness, to=JobBusiness)


injector = Injector([configure])

job_business_instance = injector.get(JobBusiness)
job_business_instance.run_scheduler()


def map_actions(controller):
    inject_controller = injector.get(controller)

    rule_attribute = 'rule'
    methods_attribute = 'method_types'

    actions = [func for func in dir(controller)
               if callable(getattr(controller, func)) and hasattr(getattr(controller, func), rule_attribute)]

    for action_name in actions:
        action = getattr(controller, action_name)
        if hasattr(action, rule_attribute) and hasattr(action, methods_attribute):
            route_callable = getattr(inject_controller, action_name)
            controller.blue_print.add_url_rule(route_callable.rule,
                                               view_func=route_callable,
                                               methods=route_callable.method_types)

    app.register_blueprint(controller.blue_print, url_prefix=f'/{controller.blue_print.name}')

jobs_started = False

def create_app():

    from .main.routes import main
    app.register_blueprint(main)

    map_actions(DrugController)
    map_actions(EnzymeController)
    map_actions(PathwayController)
    map_actions(TargetController)
    map_actions(SimilarityController)
    map_actions(DrugEmbeddingController)
    map_actions(ReductionController)
    map_actions(TrainingController)
    map_actions(JobController)

    return app
