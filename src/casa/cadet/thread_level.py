import numpy as np

def get_aspect(result):    
    if not result:
        return (None, None, None, None)
    
    entity_probs = result["entity_probs"]    
    if entity_probs.max() < 0.4:
        entity = None        
        ent_prob = None
    else:
        entity = result["entity"][np.argmax(entity_probs)]
        ent_prob = entity_probs[np.argmax(entity_probs)]
        
    service_probs = result["service_probs"]
    if service_probs.max() < 0.4:
        service = None        
        srv_prob = None
    else:
        service = result["service"][np.argmax(service_probs)]
        srv_prob = service_probs[np.argmax(service_probs)]
    return (entity, service, ent_prob, srv_prob)

def get_thread_aspect(thread_x):
    entity = None
    service = None
    entity_prob = None
    service_prob = None

    if thread_x.main:
        cadet_result = getattr(thread_x.main, "cadet_title", None)
        title_entity, title_service, title_ep, title_sp = get_aspect(cadet_result)    
        cadet_result = getattr(thread_x.main, "cadet_result", None)
        main_entity, main_service, main_ep, main_sp = get_aspect(cadet_result)
        # print("title: ", title_entity, title_service)
        # print("main: ", main_entity, main_service)
        entity = title_entity or main_entity
        service = title_service or main_service
        entity_prob = title_ep or main_ep
        service_prob = title_sp or main_sp

    for reply_x in thread_x.replies:
        cadet_result = getattr(reply_x, "cadet_result", None)
        reply_entity, reply_service, reply_ep, reply_sp = get_aspect(cadet_result)
        entity = entity or reply_entity
        service = service or reply_service
        entity_prob = entity_prob or reply_ep
        service_prob = service_prob or reply_sp
        # print("reply: ", reply_entity, reply_service)
        if entity and service:
            break
    return (entity, service, entity_prob, service_prob)