from app.cli import build_parser

def test_train_parser():
    a=build_parser().parse_args(['train','--epochs','2']); assert a.command=='train' and a.epochs==2

def test_generate_parser():
    a=build_parser().parse_args(['generate','--product','Rosewater mist','--tone','elegant']); assert a.product=='Rosewater mist'
