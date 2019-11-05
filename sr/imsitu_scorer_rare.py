import torch
import json

class imsitu_scorer():
    def __init__(self, encoder,topk, nref, image_group):
        self.score_cards = {}
        self.topk = topk
        self.nref = nref
        self.encoder = encoder
        self.hico_pred = None
        self.hico_target = None
        self.image_group = image_group
        self.topk_issue = {}

    def clear(self):
        self.score_cards = {}

    def add_point_noun(self, gt_verbs, labels_predict, gt_labels):

        batch_size = gt_verbs.size()[0]
        for i in range(batch_size):
            gt_verb = gt_verbs[i]
            label_pred = labels_predict[i]
            gt_label = gt_labels[i]

            gt_v = gt_verb
            role_set = self.encoder.get_role_ids(gt_v)

            new_card = {"verb":0.0, "value":0.0, "value*":0.0, "n_value":0.0, "value-all":0.0, "value-all*":0.0}

            score_card = new_card

            verb_found = False

            gt_role_count = self.encoder.get_role_count(gt_v)
            gt_role_list = self.encoder.verb2_role_dict[self.encoder.verb_list[gt_v]]
            score_card["n_value"] += gt_role_count

            all_found = True
            pred_list = []
            for k in range(0, self.encoder.get_max_role_count()):
                role_id = role_set[k]
                if role_id == len(self.encoder.role_list):
                    continue
                current_role = self.encoder.role_list[role_id]
                if current_role not in gt_role_list:
                    continue

                label_id = torch.max(label_pred[k],0)[1]
                pred_list.append(label_id.item())
                found = False
                for r in range(0,self.nref):
                    gt_label_id = gt_label[r][k]
                    if label_id == gt_label_id:
                        found = True
                        break
                if not found: all_found = False
                #both verb and at least one val found
                if found and verb_found: score_card["value"] += 1
                #at least one val found
                if found: score_card["value*"] += 1
            #both verb and all values found
            score_card["value*"] /= gt_role_count
            score_card["value"] /= gt_role_count
            if all_found and verb_found: score_card["value-all"] += 1
            #all values found
            if all_found: score_card["value-all*"] += 1

            self.score_cards.append(new_card)


    def add_point_noun_single_role(self, labels_predict, gt_labels):

        batch_size = gt_labels.size()[0]
        for i in range(batch_size):
            label_pred = labels_predict[i]
            gt_label = gt_labels[i]

            new_card = {"verb":0.0, "value":0.0, "value*":0.0, "n_value":0.0, "value-all":0.0, "value-all*":0.0}

            score_card = new_card

            verb_found = False

            score_card["n_value"] += 1

            all_found = True

            label_id = torch.max(label_pred,0)[1]
            found = False
            for r in range(0,self.nref):
                gt_label_id = gt_label[r]
                if label_id == gt_label_id:
                    found = True
                    break
            if not found: all_found = False
            #both verb and at least one val found
            if found and verb_found: score_card["value"] += 1
            #at least one val found
            if found: score_card["value*"] += 1

            score_card["value*"] /= 1
            score_card["value"] /= 1
            if all_found and verb_found: score_card["value-all"] += 1
            #all values found
            if all_found: score_card["value-all*"] += 1

            self.score_cards.append(new_card)


    def add_point_noun_log(self, img_id, gt_verbs, labels_predict, gt_labels):

        batch_size = gt_verbs.size()[0]
        for i in range(batch_size):
            _image = img_id[i]
            gt_verb = gt_verbs[i]
            label_pred = labels_predict[i]
            gt_label = gt_labels[i]

            gt_v = gt_verb

            if img_id is not None and _image in self.image_group: sc_key = (gt_v, self.image_group[_image])
            else: sc_key = (gt_v, "")

            if sc_key not in self.score_cards:
                new_card = {"verb":0.0, "n_image":0.0, "value":0.0, "value*":0.0, "n_value":0.0, "value-all":0.0, "value-all*":0.0}
                self.score_cards[sc_key] = new_card

            score_card = self.score_cards[sc_key]
            score_card["n_image"] += 1

            verb_found = False

            gt_role_count = self.encoder.get_role_count(gt_v)
            gt_role_list = self.encoder.verb2_role_dict[self.encoder.verb_list[gt_v]]
            score_card["n_value"] += gt_role_count



            all_found = True
            pred_situ = []
            for k in range(0, gt_role_count):


                label_id = torch.max(label_pred[k],0)[1]

                found = False
                pred_situ.append({gt_role_list[k] : self.encoder.all_words[self.encoder.labelid2nlword[self.encoder.label_list[label_id]]]})



                for r in range(0,self.nref):
                    gt_label_id = gt_label[r][k]

                    if label_id == gt_label_id:
                        found = True
                        break

                if not found:
                    all_found = False

                #both verb and at least one val found
                if found and verb_found: score_card["value"] += 1
                #at least one val found
                if found: score_card["value*"] += 1

            #both verb and all values found
            score_card["value*"] /= gt_role_count
            score_card["value"] /= gt_role_count
            if all_found and verb_found: score_card["value-all"] += 1
            #all values found
            if all_found:
                score_card["value-all*"] += 1

            self.score_cards.append(new_card)

    def add_point_noun_log_topk(self, img_id, gt_verbs, labels_predict, gt_labels):

        batch_size = gt_verbs.size()[0]
        for i in range(batch_size):
            imgid = img_id[i]
            gt_verb = gt_verbs[i]
            label_pred = labels_predict[i]
            gt_label = gt_labels[i]

            gt_v = gt_verb

            new_card = {"verb":0.0, "value":0.0, "value*":0.0, "n_value":0.0, "value-all":0.0, "value-all*":0.0}

            score_card = new_card

            verb_found = False

            gt_role_count = self.encoder.get_role_count(gt_v)
            gt_role_list = self.encoder.verb2_role_dict[self.encoder.verb_list[gt_v]]
            score_card["n_value"] += gt_role_count

            if self.write_to_file:
                self.all_res[imgid] = {'gtv': self.encoder.verb_list[gt_v], 'role_pred':[], 'all_correct': True}

            all_found = True
            pred_situ = []
            for k in range(0, gt_role_count):
                if self.write_to_file:
                    all_val = self.encoder.verb_list[gt_v] + '_' + gt_role_list[k]
                    if all_val not in self.all_verb_role:
                        self.all_verb_role[all_val] = 1
                    else:
                        self.all_verb_role[all_val] += 1

                #label_id = torch.max(label_pred[k],0)[1]

                sorted_idx = torch.sort(label_pred[k], 0, True)[1]

                found = False

                for r in range(0,self.nref):
                    gt_label_id = gt_label[r][k]

                    role_found = (torch.sum(sorted_idx[0:5] == gt_label_id) == 1)

                    if role_found:
                        found = True
                        break

                if not found:
                    all_found = False

                #both verb and at least one val found
                if found and verb_found: score_card["value"] += 1
                #at least one val found
                if found: score_card["value*"] += 1

            #both verb and all values found
            score_card["value*"] /= gt_role_count
            score_card["value"] /= gt_role_count
            if all_found and verb_found: score_card["value-all"] += 1
            #all values found
            if all_found:
                score_card["value-all*"] += 1
                if self.write_to_file:
                    self.vall_all_correct[imgid] = pred_situ
            else:
                if self.write_to_file:
                    self.value_all_dict[imgid] = pred_situ

            self.score_cards.append(new_card)

    def add_point_verb_only_eval(self, img_id, verb_predict, gt_verbs):
        #encoded predictions should be batch x verbs x values #assumes the are the same order as the references
        #encoded reference should be batch x 1+ references*roles,values (sorted)

        batch_size = verb_predict.size()[0]
        for i in range(batch_size):
            verb_pred = verb_predict[i]
            gt_verb = gt_verbs[i]
            current_id = img_id[i]

            #print('check sizes:', verb_pred.size(), gt_verb.size(), label_pred.size(), gt_label.size())
            sorted_idx = torch.sort(verb_pred, 0, True)[1]

            gt_v = gt_verb



            new_card = {"verb":0.0, "value":0.0, "value*":0.0, "n_value":0.0, "value-all":0.0, "value-all*":0.0}
            if self.write_to_file:
                self.all_res[current_id] = {'gtv': self.encoder.verb_list[gt_verb.item()],
                                            'predicted' : self.encoder.verb_list[sorted_idx[0]]}


            score_card = new_card

            verb_found = (torch.sum(sorted_idx[0:self.topk] == gt_v) == 1)
            if verb_found:
                score_card["verb"] += 1
                if self.write_to_file:
                    self.pass_list.append(current_id)
                    self.all_res[current_id]['found'] = 0

            self.score_cards.append(score_card)

    def add_point_both(self, verb_predict, gt_verbs, labels_predict, gt_labels):

        batch_size = gt_verbs.size()[0]
        for i in range(batch_size):
            verb_pred = verb_predict[i]
            gt_verb = gt_verbs[i]
            label_pred = labels_predict[i]
            gt_label = gt_labels[i]

            gt_v = gt_verb
            role_set = self.encoder.get_role_ids(gt_v)

            new_card = {"verb":0.0, "value":0.0, "value*":0.0, "n_value":0.0, "value-all":0.0, "value-all*":0.0}

            score_card = new_card

            sorted_idx = torch.sort(verb_pred, 0, True)[1]

            verb_found = (torch.sum(sorted_idx[0:self.topk] == gt_v) == 1)
            if verb_found:
                score_card["verb"] += 1

            verb_found = False

            gt_role_count = self.encoder.get_role_count(gt_v)
            gt_role_list = self.encoder.verb2_role_dict[self.encoder.verb_list[gt_v]]
            score_card["n_value"] += gt_role_count

            all_found = True
            pred_list = []
            for k in range(0, self.encoder.get_max_role_count()):
                role_id = role_set[k]
                if role_id == len(self.encoder.role_list):
                    continue
                current_role = self.encoder.role_list[role_id]
                if current_role not in gt_role_list:
                    continue

                label_id = torch.max(label_pred[k],0)[1]
                pred_list.append(label_id.item())
                found = False
                for r in range(0,self.nref):
                    gt_label_id = gt_label[r][k]
                    if label_id == gt_label_id:
                        found = True
                        break
                if not found: all_found = False
                #both verb and at least one val found
                if found and verb_found: score_card["value"] += 1
                #at least one val found
                if found: score_card["value*"] += 1
            #both verb and all values found
            score_card["value*"] /= gt_role_count
            score_card["value"] /= gt_role_count
            if all_found and verb_found: score_card["value-all"] += 1
            #all values found
            if all_found: score_card["value-all*"] += 1

            self.score_cards.append(new_card)

    def combine(self, rv, card):
        for (k,v) in card.items(): rv[k] += v

    def get_average_results(self, groups = []):
        #average across score cards.
        rv = {"verb":0, "value":0 , "value*":0 , "value-all":0, "value-all*":0}
        agg_cards = {}
        for (key, card) in self.score_cards.items():
            (v,g) = key
            if len(groups) == 0 or g in groups:
                if v not in agg_cards:
                    new_card = {"verb":0.0, "n_image":0.0, "value":0.0, "value*":0.0, "n_value":0.0, "value-all":0.0, "value-all*":0.0}
                    agg_cards[v] = new_card
                self.combine(agg_cards[v], card)
        nverbs = len(agg_cards)
        for (v, card) in agg_cards.items():
            img = card["n_image"]
            nvalue = card["n_value"]
            rv["verb"] += card["verb"]/img
            rv["value-all"] += card["value-all"]/img
            rv["value-all*"] += card["value-all*"]/img
            rv["value"] += card["value"]/nvalue
            rv["value*"] += card["value*"]/nvalue

        rv["verb"] /= nverbs
        rv["value-all"] /= nverbs
        rv["value-all*"] /= nverbs
        rv["value"] /= nverbs
        rv["value*"] /= nverbs

        return rv